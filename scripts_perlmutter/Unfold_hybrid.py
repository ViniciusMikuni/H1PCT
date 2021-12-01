import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import json
from omnifold_hybrid import  Multifold,Scaler
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


parser.add_argument('--niter', type=int,default=5, help='Number of omnifold iterations')
parser.add_argument('--nhead', type=int,default=2, help='Number of heads for multi-head PCT')
parser.add_argument('--nevts', type=float,default=40e6, help='Number of events to load')
parser.add_argument('--unfold', action='store_true', default=False,help='Train omnifold')
parser.add_argument('--closure', action='store_true', default=False,help='Train omnifold for a closure test using simulation')
parser.add_argument('--pct', action='store_true', default=False,help='Use PCT as the backbone model')
parser.add_argument('--verbose', action='store_true', default=False,help='Display additional information during training')

flags = parser.parse_args()


if flags.verbose:
    print(80*'#')
    print("Total hvd size {}, rank: {}, local size: {}, local rank{}".format(hvd.size(), hvd.rank(), hvd.local_size(), hvd.local_rank()))
    print(80*'#')


mc_names = ['Rapgap']
#mc_names = ['Django']
mc_tags = ['nominal']

var_names = [
    'jet_pt','jet_eta','jet_phi',
    'Q2',
    #'e_px','e_py','e_pz',
    'jet_ncharged',  'jet_charge', 'jet_ptD',
    'jet_tau10', 'jet_tau15', 'jet_tau20']

gen_names = [
    'genjet_pt','genjet_eta','genjet_phi',
    'gen_Q2',
    #'gene_px','gene_py','gene_pz',
    'gen_jet_ncharged','gen_jet_charge', 'gen_jet_ptD',
    'gen_jet_tau10', 'gen_jet_tau15', 'gen_jet_tau20']

global_names = ['Q2',
                #'e_px','e_py','e_pz',
                'jet_pt','jet_eta','jet_phi']

if flags.pct:
    var_names = ['jet_part_eta','jet_part_phi','jet_part_pt','jet_part_charge']

            
nevts=int(flags.nevts)
data_name = 'data'
if flags.closure:
    data_name = 'Djangoh_nominal'

for name,tag in zip(mc_names,mc_tags):
    mc_name = "{}_{}".format(name,tag)
    if mc_name == data_name:continue
    
    global_vars = {}
    mc = h5.File(os.path.join(flags.data_folder,"{}.h5".format(mc_name)),'r')
    data = h5.File(os.path.join(flags.data_folder,"{}.h5".format(data_name)),'r')


    if flags.closure:
        ntest = int(2e6)
        data_vars = np.concatenate([np.expand_dims(data[var][hvd.rank():ntest:hvd.size()],-1) for var in var_names],-1)    
        global_vars['data'] = np.concatenate([np.expand_dims(data[var][hvd.rank():ntest:hvd.size()],-1) for var in global_names],-1)

        weights_data = data['wgt'][hvd.rank():ntest:hvd.size()]
        pass_reco = data['pass_reco'][hvd.rank():ntest:hvd.size()] #pass reco selection
        weights_data = weights_data[pass_reco==1]
        data_vars = data_vars[pass_reco==1]
        global_vars['data'] =global_vars['data'][pass_reco==1]
        weights_data = weights_data/np.average(weights_data)
 
        
    mc_reco = np.concatenate([np.expand_dims(mc[var][hvd.rank():nevts:hvd.size()],-1) for var in var_names],-1)
    global_vars['reco'] = np.concatenate([np.expand_dims(mc[var][hvd.rank():nevts:hvd.size()],-1) for var in global_names],-1)    
    mc_gen = np.concatenate([np.expand_dims(mc[var][hvd.rank():nevts:hvd.size()],-1) for var in gen_names],-1)

    #Same preprocessing over all tasks
    mean,std = Scaler(mc,global_names)
    global_vars['reco'] = (global_vars['reco']-mean)/std
    global_vars['data'] = (global_vars['data']-mean)/std
    
    mean,std = Scaler(mc,gen_names)
    mc_gen = (mc_gen-mean)/std
    if not flags.pct:
        mc_reco = (mc_reco-mean)/std
        data_vars = (data_vars-mean)/std

    weights_MC_sim = mc['wgt'][hvd.rank():nevts:hvd.size()]
    pass_reco = mc['pass_reco'][hvd.rank():nevts:hvd.size()] #pass fiducial selection
    pass_truth = mc['pass_truth'][hvd.rank():nevts:hvd.size()] #pass gen region definition
    del mc
    del data



    weights_MC_sim = weights_MC_sim/np.average(weights_MC_sim)
    weights_MC_sim *= 1.0*data_vars.shape[0]/weights_MC_sim[pass_reco==1].shape[0]

    if flags.pct:
        mc_reco[:,0,0][pass_reco==0] = -10
    else:
        mc_reco[:,0][pass_reco==0] = -10

    mc_gen[:,0][pass_truth==0] = -10
    
    if flags.verbose:
        print(80*'#')
        print("Events passing reco cuts: {} Events passing gen cuts: {} ".format(np.sum(pass_reco==1),np.sum(pass_truth==1)))
        print(80*'#')
        
    version = 'closure' if flags.closure else 'baseline'
    mfold = Multifold(
        niter=flags.niter,
        nhead=flags.nhead,
        pct=flags.pct,
        version = version,
        global_vars=global_vars,
        nglobal = len(global_names),
        nevts=nevts,
        verbose = flags.verbose,
    )


    print(mc_reco.shape,data_vars.shape)
    mfold.mc_gen = mc_gen
    mfold.mc_reco = mc_reco
    mfold.data = data_vars

    if flags.closure:
        mfold.Preprocessing(weights_mc=weights_MC_sim,weights_data=weights_data)
    else:
        mfold.Preprocessing(weights_mc=weights_MC_sim)

    mfold.Unfold()
