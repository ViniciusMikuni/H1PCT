import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import json
from omnifold import  Multifold,Scaler,LoadJson
import h5py as h5
import os
import horovod.tensorflow.keras as hvd
import tensorflow as tf
import tensorflow.keras.backend as K



hvd.init()
# Horovod: pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

# tf.random.set_seed(1234)
# np.random.seed(1234)

parser = argparse.ArgumentParser()

parser.add_argument('--data_folder', default='/pscratch/sd/v/vmikuni/H1', help='Folder containing data and MC files')
parser.add_argument('--mode', default='hybrid', help='[standard/hybrid/PCT]')
parser.add_argument('--config', default='config_general.json', help='Basic config file containing general options')
parser.add_argument('--nevts', type=float,default=25e6, help='Number of events to load')
parser.add_argument('--closure', action='store_true', default=False,help='Train omnifold for a closure test using simulation')
parser.add_argument('--nstrap', type=int,default=0, help='Unique id for bootstrapping')
parser.add_argument('--verbose', action='store_true', default=False,help='Display additional information during training')

flags = parser.parse_args()

if flags.verbose:
    print(80*'#')
    print("Total hvd size {}, rank: {}, local size: {}, local rank{}".format(hvd.size(), hvd.rank(), hvd.local_size(), hvd.local_rank()))
    print(80*'#')

opt=LoadJson(flags.config)

if flags.closure:
    mc_names = ['Rapgap_nominal']
    #mc_names = ['Djangoh_nominal'] 
else:
    mc_names = opt['MC_NAMES']

if flags.mode == 'standard':
    var_names = opt['VAR_MLP']
    gen_names = opt['VAR_MLP_GEN']
    global_names_reco = []
    global_names_gen = []
elif flags.mode == 'hybrid':
    var_names = opt['VAR_PCT']
    gen_names = opt['VAR_MLP_GEN']
    global_names_reco = opt['GLOBAL_RECO']
    global_names_gen = []
elif flags.mode == 'PCT':
    var_names = opt['VAR_PCT']
    gen_names = opt['VAR_PCT_GEN']
    global_names_reco = opt['GLOBAL_RECO']
    global_names_gen = opt['GLOBAL_GEN']
else:
    raise ValueError("ERROR: Running mode not found!")

            
nevts=int(flags.nevts)
data_name = 'data'
if flags.closure:
    data_name = 'Djangoh_nominal'
    #data_name = 'Rapgap_nominal'

    
for mc_name in mc_names:
    
    global_vars = {}
    mc = h5.File(os.path.join(flags.data_folder,"{}.h5".format(mc_name)),'r')
    data = h5.File(os.path.join(flags.data_folder,"{}.h5".format(data_name)),'r')


    if flags.closure:
        ntest = int(5e6) #about same number of data events after reco selection
        data_vars = np.concatenate([np.expand_dims(data[var][hvd.rank():ntest:hvd.size()],-1) for var in var_names],-1)
        weights_data = data['wgt'][hvd.rank():ntest:hvd.size()]
        pass_reco = data['pass_reco'][hvd.rank():ntest:hvd.size()] #pass reco selection
        
        if flags.mode != 'standard':
            global_vars['data'] = np.concatenate([np.expand_dims(data[var][hvd.rank():ntest:hvd.size()],-1) for var in global_names_reco],-1)
            global_vars['data'] =global_vars['data'][pass_reco==1]
        else:
            global_vars['data'] = np.array([])

        
        weights_data = weights_data[pass_reco==1]
        data_vars = data_vars[pass_reco==1]        
        weights_data = weights_data/np.average(weights_data)
    else:
        data_vars = np.concatenate([np.expand_dims(data[var][hvd.rank()::hvd.size()],-1) for var in var_names],-1)
        if flags.mode != 'standard':
            global_vars['data'] = np.concatenate([np.expand_dims(data[var][hvd.rank()::hvd.size()],-1) for var in global_names_reco],-1)
        else:
            global_vars['data'] = np.array([])
        if flags.nstrap>0:
            if flags.verbose:
                print(80*"#")
                print("Running booststrap with ID: {}".format(flags.nstrap))
                np.random.seed(flags.nstrap*hvd.rank())
                print(80*"#")
            weights_data = np.random.poisson(1,data_vars.shape[0])
        else:
            weights_data = np.ones(data_vars.shape[0])
            
        
 
        
    mc_reco = np.concatenate([np.expand_dims(mc[var][hvd.rank():nevts:hvd.size()],-1) for var in var_names],-1)
    mc_gen = np.concatenate([np.expand_dims(mc[var][hvd.rank():nevts:hvd.size()],-1) for var in gen_names],-1)    
    if flags.mode != 'standard':
        global_vars['reco'] = np.concatenate([np.expand_dims(mc[var][hvd.rank():nevts:hvd.size()],-1) for var in global_names_reco],-1)
        if flags.mode == "PCT":
            global_vars['gen'] = np.concatenate([np.expand_dims(mc[var][hvd.rank():nevts:hvd.size()],-1) for var in global_names_gen],-1)
        else:
            pass
            # tau_idx = [4,5,6] #CAUTION!!! IF THAT CHANGES REMEMBER TO CHANGE THIS LINE TOO
            # for idx in tau_idx:
            #     mc_gen[:,idx] = np.ma.log(mc_gen[:,idx]).filled(0)

    weights_MC_sim = mc['wgt'][hvd.rank():nevts:hvd.size()]
    pass_reco = mc['pass_reco'][hvd.rank():nevts:hvd.size()] #pass fiducial selection
    pass_truth = mc['pass_truth'][hvd.rank():nevts:hvd.size()] #pass gen region definition

    
    #Same preprocessing over all tasks
    if flags.mode == "hybrid":
        mean,std = Scaler(mc,global_names_reco)
        global_vars['reco'] = (global_vars['reco']-mean)/std
        global_vars['data'] = (global_vars['data']-mean)/std        
        mean,std = Scaler(mc,gen_names)
        mc_gen = (mc_gen-mean)/std
        
        mc_reco[:,0,0][pass_reco==0] = -10
        mc_gen[:,0][pass_truth==0] = -10
        
    if flags.mode == 'PCT':
        mean,std = Scaler(mc,global_names_gen)
        global_vars['reco'] = (global_vars['reco']-mean)/std
        global_vars['data'] = (global_vars['data']-mean)/std
        global_vars['gen'] = (global_vars['gen']-mean)/std

        mc_reco[:,0,0][pass_reco==0] = -10
        mc_gen[:,0,0][pass_truth==0] = -10
        
    if flags.mode == 'standard':            
        mean,std = Scaler(mc,gen_names)
        mc_gen = (mc_gen-mean)/std
        mc_reco = (mc_reco-mean)/std
        data_vars = (data_vars-mean)/std
        
        mc_reco[:,0][pass_reco==0] = -10
        mc_gen[:,0][pass_truth==0] = -10
        
    del mc
    del data

    weights_MC_sim = weights_MC_sim/np.sum(weights_MC_sim[pass_reco==1])
    weights_MC_sim *= 1.0*data_vars.shape[0]
    
    if flags.verbose:
        print(80*'#')
        print("Events passing reco cuts: {} Events passing gen cuts: {} ".format(np.sum(pass_reco==1),np.sum(pass_truth==1)))
        print(80*'#')

    version = mc_name
    if flags.closure:
        version += '_closure'
        
    K.clear_session()
    mfold = Multifold(
        mode=flags.mode,
        version = version,
        nevts=nevts,
        nstrap=flags.nstrap,
        verbose = flags.verbose,        
    )

    mfold.mc_gen = mc_gen
    mfold.mc_reco = mc_reco
    mfold.data = data_vars
    mfold.global_vars=global_vars

    
    mfold.Preprocessing(weights_mc=weights_MC_sim,weights_data=weights_data)    
    mfold.Unfold()
