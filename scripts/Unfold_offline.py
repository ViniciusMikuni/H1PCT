import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import argparse
import json
from dataloader2 import get_Dataframe, applyCut, applyCutsJets
from unfold import  Multifold
import h5py as h5
import os
os.environ['CUDA_VISIBLE_DEVICES']="3"

import tensorflow as tf
import tensorflow.keras.backend as K
# tf.random.set_seed(1234)
# np.random.seed(1234)

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

parser = argparse.ArgumentParser()

parser.add_argument('--data_folder', default='/clusterfs/ml4hep/vmikuni/H1/jet_subs/h5', help='Folder containing data and MC files')

parser.add_argument('--nvars', type=int, default=10, help='Number of distributions to unfold')
parser.add_argument('--niter', type=int,default=5, help='Number of omnifold iterations')
parser.add_argument('--ntrain', type=int,default=5, help='Number of independent trainings to perform')
parser.add_argument('--nevts', type=int,default=10000000, help='Number of events to load')
parser.add_argument('--reload', action='store_true', default=False,help='Redo the data preparation steps')
parser.add_argument('--unfold', action='store_true', default=False,help='Train omnifold')
parser.add_argument('--closure', action='store_true', default=False,help='Train omnifold for a closure test using simulation')
parser.add_argument('--load_model', action='store_true', default=False,help='Load pretrained weights')
parser.add_argument('--pct', action='store_true', default=False,help='Use PCT as the backbone model')

flags = parser.parse_args()


mc_names = ['Rapgap']
#mc_names = ['Django']
mc_tags = ['nominal']

var_names = ['jet_pt','jet_eta','jet_phi','jet_ncharged', 'Q2',
             'jet_charge', 'jet_ptD','jet_tau10', 'jet_tau15', 'jet_tau20']

gen_names = ['genjet_pt','genjet_eta','genjet_phi','gen_jet_ncharged','gen_Q2',
              'gen_jet_charge', 'gen_jet_ptD','gen_jet_tau10', 'gen_jet_tau15', 'gen_jet_tau20']

if flags.pct:
    var_names = ['jet_part_eta','jet_part_phi','jet_part_pt','jet_part_charge']




if flags.reload:
    # data = get_Dataframe(flags.data_folder, name='Data', tag=data_tag, pct=flags.pct,verbose=True)
    # #print(data.keys())
    # data = applyCutsJets(data,verbose=True,pct=flags.pct)
    # data.to_pickle(os.path.join(flags.data_folder,'pkl','data.pkl' if flags.pct == False else 'data_pct.pkl'))
    
    for name,tag in zip(mc_names,mc_tags):
        mc = get_Dataframe(flags.data_folder, name=name, tag=tag, pct=flags.pct,verbose=True)
        mc   = applyCutsJets(mc, isMC=True,verbose=True,pct=flags.pct)
        #if flags.pct:tag+='_pct'
        with h5.File(os.path.join('/clusterfs/ml4hep/vmikuni/H1/jet_subs/','h5',"{}_{}.h5".format(name,tag)),'w') as fh5:
            for key in mc:
                print(key)
                if 'part' in key:
                    feat = np.array([ entry for entry in mc[key].to_numpy()])
                    feat = feat.reshape((len(feat),20)).astype(float)                    
                    dset = fh5.create_dataset(key, data=feat)
                else:
                    dset = fh5.create_dataset(key, data=mc[key].to_numpy(dtype=np.float32))
            

data_name = 'data'
if flags.closure:
    data_name = 'Djangoh_nominal'

for name,tag in zip(mc_names,mc_tags):
    mc_name = "{}_{}".format(name,tag)
    if mc_name == data_name:continue
    
    Q2 = {}
    mc = h5.File(os.path.join(flags.data_folder,"{}.h5".format(mc_name)),'r')    

    data = h5.File(os.path.join(flags.data_folder,"{}.h5".format(data_name)),'r')
    data_vars = np.concatenate([np.expand_dims(data[var][:flags.nevts],-1) for var in var_names],-1)    
    Q2['data'] =  np.ma.log(data['Q2'][:flags.nevts]).filled(0)

    if flags.closure:
        weights_data = data['wgt'][:flags.nevts]
        pass_reco = data['pass_reco'][:flags.nevts] #pass reco selection
        weights_data = weights_data[pass_reco==1]
        data_vars = data_vars[pass_reco==1]
        Q2['data'] =Q2['data'][pass_reco==1]
        weights_data = weights_data/np.average(weights_data)


    mc_reco = np.concatenate([np.expand_dims(mc[var][:flags.nevts],-1) for var in var_names],-1)
    Q2['reco'] = np.ma.log(mc['Q2'][:flags.nevts]).filled(0)
    mc_gen = np.concatenate([np.expand_dims(mc[var][:flags.nevts],-1) for var in gen_names],-1)

    weights_MC_sim = mc['wgt'][:flags.nevts]
    pass_reco = mc['pass_reco'][:flags.nevts] #pass fiducial selection
    pass_truth = mc['pass_truth'][:flags.nevts] #pass gen region definition
    del mc
    del data



    weights_MC_sim = weights_MC_sim/np.average(weights_MC_sim[pass_reco==1])
    weights_MC_sim *= 1.0*data_vars.shape[0]/weights_MC_sim[pass_reco==1].shape[0]

    if flags.pct:
        mc_reco[:,0,0][pass_reco==0] = -10
    else:
        mc_reco[:,0][pass_reco==0] = -10

    mc_gen[:,0][pass_truth==0] = -10

    print("Events passing reco cuts: {} Events passing gen cuts: {} ".format(np.sum(pass_reco==1),np.sum(pass_truth==1)))

    version = 'closure' if flags.closure else 'baseline'
    mfold = Multifold(nvars=flags.nvars,
                      niter=flags.niter,
                      pct=flags.pct,
                      version = version,
                      Q2=Q2,
    )

    mfold.mc_gen = mc_gen
    mfold.mc_reco = mc_reco
    mfold.data = data_vars
    if flags.closure:
        mfold.Preprocessing(weights_mc=weights_MC_sim,weights_data=weights_data)
    else:
        mfold.Preprocessing(weights_mc=weights_MC_sim)

    mfold.Unfold()
