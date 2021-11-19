import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import argparse
import json
from unfold_hvd import  Multifold
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

#parser.add_argument('--data_folder', default='/global/cfs/cdirs/m1759/vmikuni/H1/h5', help='Folder containing data and MC files')

parser.add_argument('--data_folder', default='/pscratch/sd/v/vmikuni/H1', help='Folder containing data and MC files')

parser.add_argument('--nvars', type=int, default=10, help='Number of distributions to unfold')
parser.add_argument('--niter', type=int,default=5, help='Number of omnifold iterations')
parser.add_argument('--nevts', type=float,default=50e6, help='Number of events to train per sample')
parser.add_argument('--ntrain', type=int,default=5, help='Number of independent trainings to perform')
parser.add_argument('--reload', action='store_true', default=False,help='Redo the data preparation steps')
parser.add_argument('--unfold', action='store_true', default=False,help='Train omnifold')
parser.add_argument('--closure', action='store_true', default=False,help='Train omnifold for a closure test using simulation')
parser.add_argument('--plot', action='store_true', default=False,help='Display validation plots along the code')
parser.add_argument('--pct', action='store_true', default=False,help='Use PCT as the backbone model')

flags = parser.parse_args()


mc_names = ['Djangoh','Rapgap']
mc_tags = ['nominal','nominal']

var_names = ['jet_pt','jet_eta','jet_phi','jet_ncharged','Q2',
             'jet_charge', 'jet_ptD','jet_tau10', 'jet_tau15', 'jet_tau20']

gen_names = ['genjet_pt','genjet_eta','genjet_phi','gen_jet_ncharged','gen_Q2',
              'gen_jet_charge', 'gen_jet_ptD','gen_jet_tau10', 'gen_jet_tau15', 'gen_jet_tau20']

if flags.pct:
    var_names = ['jet_part_eta','jet_part_phi','jet_part_pt','jet_part_charge']
    
    gen_names = ['gen_jet_part_eta','gen_jet_part_phi','gen_jet_part_pt','gen_jet_part_charge']

nevts=int(flags.nevts)
data_name = 'data'
if flags.closure:
    data_name = 'Djangoh_nominal'


for name,tag in zip(mc_names,mc_tags):
    mc_name = "{}_{}".format(name,tag)
    if mc_name == data_name:continue
    
    Q2 = {}    
    mc = h5.File(os.path.join(flags.data_folder,"{}.h5".format(mc_name)),'r')

    data = h5.File(os.path.join(flags.data_folder,"{}.h5".format(data_name)),'r')
    data_vars = np.concatenate([np.expand_dims(data[var][hvd.rank():nevts:hvd.size()],-1) for var in var_names],-1)
    Q2['data'] =  np.ma.log(data['Q2'][hvd.rank():nevts:hvd.size()]).filled(-1)

    if flags.closure:
        weights_data = data['wgt'][hvd.rank():nevts:hvd.size()]
        pass_reco = data['pass_reco'][hvd.rank():nevts:hvd.size()] #pass reco selection
        weights_data = weights_data[pass_reco==1]
        data_vars = data_vars[pass_reco==1]
        Q2['data'] =Q2['data'][pass_reco==1]
        weights_data = weights_data/np.average(weights_data)
        
    if flags.pct:
        jet_pt = np.expand_dims(data['jet_pt'][hvd.rank():nevts:hvd.size()],-1)
        jet_pt = jet_pt[pass_reco==1]
        data_vars[:,:,2]*=jet_pt/100
        
    mc_reco = np.concatenate([np.expand_dims(mc[var][hvd.rank():nevts:hvd.size()],-1) for var in var_names],-1)
    mc_gen = np.concatenate([np.expand_dims(mc[var][hvd.rank():nevts:hvd.size()],-1) for var in gen_names],-1)

    Q2['reco'] = np.ma.log(mc['Q2'][hvd.rank():nevts:hvd.size()]).filled(-1)
    Q2['gen'] = np.ma.log(mc['gen_Q2'][hvd.rank():nevts:hvd.size()]).filled(-1)
    
    if flags.pct:
        jet_pt = np.expand_dims(mc['jet_pt'][hvd.rank():nevts:hvd.size()],-1)
        gen_jet_pt = np.expand_dims(mc['genjet_pt'][hvd.rank():nevts:hvd.size()],-1)
        mc_reco[:,:,2] *=jet_pt/100
        mc_gen[:,:,2] *=gen_jet_pt/100
        del jet_pt
        del gen_jet_pt


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
        
    print("Events passing reco cuts: {} Events passing gen cuts: {} ".format(np.sum(pass_reco==1),np.sum(pass_truth==1)))
    print(np.sum(pass_reco==1)/np.sum(pass_truth==1))

    version = 'closure' if flags.closure else 'baseline'
    mfold = Multifold(nvars=flags.nvars,
                      niter=flags.niter,
                      pct=flags.pct,
                      Q2=Q2,
                      version = version
    )
    
    mfold.mc_gen = mc_gen
    mfold.mc_reco = mc_reco
    mfold.data = data_vars
    if flags.closure:
        mfold.Preprocessing(weights_mc=weights_MC_sim,weights_data=weights_data)
    else:
        mfold.Preprocessing(weights_mc=weights_MC_sim)

    
    mfold.Unfold()

