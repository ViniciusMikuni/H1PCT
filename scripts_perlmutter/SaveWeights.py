import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.font_manager import FontProperties
import argparse
import os
import sys
import h5py as h5
from omnifold import  Multifold, Scaler, LoadJson
import tensorflow as tf
import tensorflow.keras.backend as K
sys.path.append('../')
import shared.options as opt
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')


class MCInfo():
    def __init__(self,mc_name,N,data_folder,config,q2_int=0,use_mask=True):
        self.N = N
        self.file = h5.File(os.path.join(data_folder,"{}.h5".format(mc_name)),'r')
        self.use_mask = use_mask
        self.config = config
        self.truth_mask = self.file['pass_truth'][:self.N] #pass truth region definition
        
        if self.use_mask:
            self.mask = self.file['pass_fiducial'][:self.N] == 1 #pass fiducial region definition
            if q2_int>0:
                gen_q2 = opt.dedicated_binning['gen_Q2']
                self.mask *= ((self.file['gen_Q2'][:self.N] > gen_q2[q2_int-1]) & (self.file['gen_Q2'][:self.N] < gen_q2[q2_int]))
                #self.fiducial_mask *= np.sum(self.file['gen_jet_part_pt'][:self.N] > 0,-1) < 20        
            self.nominal_wgts = self.file['wgt'][:self.N][self.mask]
        else:
            self.nominal_wgts = self.file['wgt'][:self.N]
            
    def LoadVar(self,var):
        if self.use_mask:
            return_var = self.file[var][:self.N][self.mask]
        else:
            return_var = self.file[var][:self.N]
        if 'tau' in var:
            return_var = np.ma.log(return_var).filled(0)
        return return_var


    def ReturnWeights(self,niter,model_name,mode='hybrid'):
        mfold = self.LoadDataWeights(niter,mode)
        return self.Reweight(mfold,model_name)
    
    def LoadDataWeights(self,niter,mode='hybrid'):

        mfold = Multifold(
            mode=mode,
            nevts = self.N
        )
        
        
        if mode == 'PCT':
            var_names = self.config['VAR_PCT_GEN']
            global_names = self.config['GLOBAL_GEN']
            global_vars = np.concatenate([np.expand_dims(self.file[var][:self.N],-1) for var in global_names],-1)
            mean,std = Scaler(self.file,global_names)
            global_vars[self.truth_mask==1] = (global_vars[self.truth_mask==1] - mean)/std

        else:
            var_names = self.config['VAR_MLP_GEN']
            global_vars = np.array([[]])

        mfold.global_vars = {'reco':global_vars}
        data = np.concatenate([np.expand_dims(self.file[var][:self.N],-1) for var in var_names],-1)
        # if mode != 'PCT':
        #     tau_idx = [4,5,6] #CAUTION!!! IF THAT CHANGES REMEMBER TO CHANGE THIS LINE TOO
        #     for idx in tau_idx:
        #         data[:,idx] = np.ma.log(data[:,idx]).filled(0)
        
        if mode != "PCT":
            mean,std = Scaler(self.file,var_names)
            data[self.truth_mask==1]=(data[self.truth_mask==1]-mean)/std
            mfold.mc_gen = data
        else:
            mfold.mc_gen = [data,global_vars]
            
        mfold.PrepareModel()
        return mfold

    def Reweight(self,mfold,model_name):
        mfold.model2.load_weights(model_name)
        if self.use_mask:
            return mfold.reweight(mfold.mc_gen,mfold.model2)[self.mask]
        else:
            return mfold.reweight(mfold.mc_gen,mfold.model2)

    def LoadTrainedWeights(self,file_name):
        h5file = h5.File(file_name,'r')
        weights = h5file['wgt'][:self.N]
        if self.use_mask:
            weights=weights[self.mask]
        return weights
        
            


if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', default='/pscratch/sd/v/vmikuni/H1', help='Folder containing data and MC files')
    parser.add_argument('--mode', default='hybrid', help='Which train type to load [hybrid/standard/PCT]')
    parser.add_argument('--config', default='config_general.json', help='Basic config file containing general options')
    parser.add_argument('--out', default='/pscratch/sd/v/vmikuni/H1/weights', help='Folder to save the weights')
    parser.add_argument('--niter', type=int, default=5, help='Omnifold iteration to load')
   
    flags = parser.parse_args()
    config=LoadJson(flags.config)
   
    ###################
    #bootstrap weights#
    ###################
    base_name = "Omnifold_{}".format(flags.mode)
    mc_names = ['Rapgap_nominal']   
    for mc_name in mc_names:
        print("{}.h5".format(mc_name))    
        mc_info = MCInfo(mc_name,int(100e6),flags.data_folder,config,use_mask=False)
        model_strap = '../weights_strap/{}_{}_iter{}_step2_strapX.h5'.format(
            base_name,mc_name,flags.niter)
        mfold = mc_info.LoadDataWeights(flags.niter,mode=flags.mode)
        for nstrap in range(69,config['NBOOTSTRAP']+1):
            print(nstrap)
            weights =  mc_info.Reweight(mfold,model_name=model_strap.replace('X',str(nstrap)))            
            with h5.File(os.path.join(flags.out,'{}_{}.h5'.format(mc_name,nstrap)),'w') as fout:
                dset = fout.create_dataset('wgt', data=weights)
            del weights
            K.clear_session()
        del mc_info

    
   # print(mc_info[mc_name].LoadTrainedWeights(os.path.join(flags.out,mc_name+'.h5'))[:100])
