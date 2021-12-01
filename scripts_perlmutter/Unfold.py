import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
from omnifold_pct import  Multifold
import h5py as h5
import os
import horovod.tensorflow.keras as hvd


import tensorflow as tf
import tensorflow.keras.backend as K

# tf.random.set_seed(1234)
#np.random.seed(1234)


hvd.init()
# Horovod: pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
    

parser = argparse.ArgumentParser()

parser.add_argument('--data_folder', default='/global/cfs/cdirs/m1759/vmikuni/H1/h5', help='Folder containing data and MC files')

#parser.add_argument('--data_folder', default='/pscratch/sd/v/vmikuni/H1', help='Folder containing data and MC files')


parser.add_argument('--niter', type=int,default=5, help='Number of omnifold iterations')
parser.add_argument('--nevts', type=float,default=50e6, help='Number of events to train per sample')
parser.add_argument('--nhead', type=int,default=5, help='Number of independent trainings to perform')
parser.add_argument('--reload', action='store_true', default=False,help='Redo the data preparation steps')
parser.add_argument('--unfold', action='store_true', default=False,help='Train omnifold')
parser.add_argument('--closure', action='store_true', default=False,help='Train omnifold for a closure test using simulation')
parser.add_argument('--plot', action='store_true', default=False,help='Display validation plots along the code')
parser.add_argument('--pct', action='store_true', default=False,help='Use PCT as the backbone model')

flags = parser.parse_args()

nevts=int(flags.nevts)
mc_names = ['Djangoh','Rapgap']
mc_tags = ['nominal','nominal']

var_names = [
    'jet_pt','jet_eta','jet_phi',
    'Q2','e_px','e_py','e_pz',
    'jet_ncharged',  'jet_charge', 'jet_ptD',
    'jet_tau10', 'jet_tau15', 'jet_tau20']

gen_names = [
    'genjet_pt','genjet_eta','genjet_phi',
    'gen_Q2', 'gene_px','gene_py','gene_pz',
    'gen_jet_ncharged','gen_jet_charge', 'gen_jet_ptD',
    'gen_jet_tau10', 'gen_jet_tau15', 'gen_jet_tau20']

if flags.pct:
    var_names = ['jet_part_eta','jet_part_phi','jet_part_pt','jet_part_charge']
    
    gen_names = ['gen_jet_part_eta','gen_jet_part_phi','gen_jet_part_pt','gen_jet_part_charge']

global_names_reco = ['Q2','e_px','e_py','e_pz']
global_names_gen = ['gen_Q2','gene_px','gene_py','gene_pz']


data_name = 'data'
if flags.closure:
    data_name = 'Djangoh_nominal'


for name,tag in zip(mc_names,mc_tags):
    mc_name = "{}_{}".format(name,tag)
    if mc_name == data_name:continue
    
    global_vars = {}    
    mc = h5.File(os.path.join(flags.data_folder,"{}.h5".format(mc_name)),'r')

    data = h5.File(os.path.join(flags.data_folder,"{}.h5".format(data_name)),'r')
    data_vars = np.concatenate([np.expand_dims(data[var][hvd.rank():nevts:hvd.size()],-1) for var in var_names],-1)
    global_vars['data'] = np.concatenate([np.expand_dims(data[var][hvd.rank():nevts:hvd.size()],-1) for var in global_names_reco],-1)

    if flags.closure:
        weights_data = data['wgt'][hvd.rank():nevts:hvd.size()]
        pass_reco = data['pass_reco'][hvd.rank():nevts:hvd.size()] #pass reco selection
        weights_data = weights_data[pass_reco==1]
        data_vars = data_vars[pass_reco==1]
        global_vars['data'] =global_vars['data'][pass_reco==1]
        weights_data = weights_data/np.average(weights_data)

        
        
    mc_reco = np.concatenate([np.expand_dims(mc[var][hvd.rank():nevts:hvd.size()],-1) for var in var_names],-1)
    global_vars['reco'] = np.concatenate([np.expand_dims(mc[var][hvd.rank():nevts:hvd.size()],-1) for var in global_names_reco],-1)
    
    mc_gen = np.concatenate([np.expand_dims(mc[var][hvd.rank():nevts:hvd.size()],-1) for var in gen_names],-1)
    global_vars['gen'] = np.concatenate([np.expand_dims(mc[var][hvd.rank():nevts:hvd.size()],-1) for var in global_names_gen],-1)    
        
    
    weights_MC_sim = mc['wgt'][hvd.rank():nevts:hvd.size()]
    pass_reco = mc['pass_reco'][hvd.rank():nevts:hvd.size()] #pass fiducial selection
    pass_truth = mc['pass_truth'][hvd.rank():nevts:hvd.size()] #pass gen region definition
    del mc
    
    weights_MC_sim = weights_MC_sim/np.average(weights_MC_sim[pass_reco==1])
    weights_MC_sim *= 1.0*data_vars.shape[0]/weights_MC_sim[pass_reco==1].shape[0]

    if flags.pct:
        mc_reco[:,0,0][pass_reco==0] = -10
        mc_gen[:,0,0][pass_truth==0] = -10
    else:
        mc_reco[:,0][pass_reco==0] = -10
        mc_gen[:,0][pass_truth==0] = -10
        

    version = 'closure' if flags.closure else 'baseline'
    mfold = Multifold(
        niter=flags.niter,
        pct=flags.pct,
        nhead=flags.nhead,
        global_vars=global_vars,
        version = version,
        nglobal = len(global_names_reco),
        nevts=2*nevts if flags.closure else nevts, #data size smaller than simulation
    )
    
    mfold.mc_gen = mc_gen
    mfold.mc_reco = mc_reco
    mfold.data = data_vars
    
    if flags.closure:
        mfold.Preprocessing(weights_mc=weights_MC_sim,weights_data=weights_data)
    else:
        mfold.Preprocessing(weights_mc=weights_MC_sim)

    
    mfold.Unfold()

