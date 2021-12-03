import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.font_manager import FontProperties
import argparse
import os
import sys
import h5py as h5
from omnifold import  Multifold, Scaler, LoadJson

sys.path.append('../')
import shared.options as opt


opt.SetStyle()

parser = argparse.ArgumentParser()

parser.add_argument('--data_folder', default='/pscratch/sd/v/vmikuni/H1', help='Folder containing data and MC files')
parser.add_argument('--weights', default='../weights', help='Folder to store trained weights')
parser.add_argument('--mode', default='hybrid', help='Which train type to load [hybrid/standard/PCT]')
parser.add_argument('--config', default='config_general.json', help='Basic config file containing general options')
parser.add_argument('--closure', action='store_true', default=False,help='Plot closure results')
parser.add_argument('--comp', action='store_true', default=False,help='Compare closure unc. from different methods')
parser.add_argument('-N',type=float,default=40e6, help='Number of events to evaluate')
parser.add_argument('--niter', type=int, default=9, help='Omnifold iteration to load')

flags = parser.parse_args()
flags.N = int(flags.N)

config=LoadJson(flags.config)

mc_names = ['Rapgap_nominal','Djangoh_nominal']
    
data_idx = 0
data_name = mc_names[data_idx]

mc_ref = mc_names[data_idx-1]
print(mc_ref)
folder = 'results'
if flags.closure:
    version = 'closure'
else:
    version = 'nominal'


gen_var_names = {
    #'jet_pt': r"$p_\mathrm{T}^\mathrm{jet}$",
    
    'gen_Q2': r"$Q^2$",
    'genjet_pt': r"$p_\mathrm{T}^\mathrm{jet}$",
    'genjet_eta':r'$\eta^\mathrm{jet}$',
    # 'genjet_phi':r'$\phi^\mathrm{jet}$',
    'gen_jet_ncharged':r'$\mathrm{N_{part}}^\mathrm{jet}$', 
    'gen_jet_charge':r'$\mathrm{Q}^\mathrm{jet}$', 
    'gen_jet_ptD':r'$p_\mathrm{T}\mathrm{D}^\mathrm{jet}$',
    'gen_jet_tau10':r'$\mathrm{log}(\tau_{1}^\mathrm{jet})$', 
    'gen_jet_tau15':r'$\mathrm{log}(\tau_{0.5}^\mathrm{jet})$',
    'gen_jet_tau20':r'$\mathrm{log}(\tau_{0}^\mathrm{jet})$',

    # 'gene_px':'e px',
    # 'gene_py':'e px',
    # 'gene_pz':'e px',
}


    
    
class MCInfo():
    def __init__(self,mc_name,N,data_folder):
        self.N = N
        self.file = h5.File(os.path.join(data_folder,"{}.h5".format(mc_name)),'r')
        self.nominal_wgts = self.file['wgt'][:self.N]
        self.fiducial_masks = self.file['pass_fiducial'][:self.N] #pass fiducial region definition
        #self.fiducial_masks *= np.sum(self.file['gen_jet_part_pt'][:self.N] > 0,-1) < 20
        self.truth_mask = self.file['pass_truth'][:self.N] #pass truth region definition
        

    def LoadDataWeights(self,niter,mode=False):

        mfold = Multifold(
            mode=mode,
            nevts = self.N
        )
        
        
        if mode == 'PCT':
            var_names = config['VAR_PCT_GEN']
            global_names = config['GLOBAL_GEN']
            global_vars = np.concatenate([np.expand_dims(self.file[var][:self.N],-1) for var in global_names],-1)
            mean,std = Scaler(self.file,global_names)
            global_vars[self.truth_mask==1] = (global_vars[self.truth_mask==1] - mean)/std

        else:
            var_names = config['VAR_MLP_GEN']
            global_vars = np.array([[]])

        mfold.global_vars = {'reco':global_vars}
        base_name = "Omnifold_{}".format(mode)            
        model_name = '{}/{}_{}_iter{}_step2.h5'.format(flags.weights,base_name,version,niter)        
        data = np.concatenate([np.expand_dims(self.file[var][:self.N],-1) for var in var_names],-1)
        
        if mode != "PCT":
            mean,std = Scaler(self.file,var_names)
            data[self.truth_mask==1]=(data[self.truth_mask==1]-mean)/std                
            mfold.mc_gen = data
        else:
            mfold.mc_gen = [data,global_vars]
            
        mfold.PrepareModel()
        mfold.model2.load_weights(model_name)
        return mfold.reweight(mfold.mc_gen,mfold.model2)

def RatioLabel(ax1):
    ax1.set_ylabel('Model unc. (%)')
    ax1.set_xlabel(gen_var_names[var])    
    ax1.axhline(y=0.0, color='r', linestyle='-')
    ax1.axhline(y=10, color='r', linestyle='--')
    ax1.axhline(y=-10, color='r', linestyle='--')
    ax1.set_ylim([-20,20])
    

    
mc_info = {}
weights_data = {}

for mc_name in mc_names:
    print("{}.h5".format(mc_name))
    
    mc_info[mc_name] = MCInfo(mc_name,flags.N,flags.data_folder)
        
    if mc_name == data_name:
        if flags.comp:
            for mode in ['hybrid','standard','PCT']:
                print(mode)
                weights_data[mode] = mc_info[mc_name].LoadDataWeights(flags.niter,mode=mode)
        else:
            weights_data[flags.mode] = mc_info[mc_name].LoadDataWeights(flags.niter,mode=flags.mode)
            

weight_data = weights_data[flags.mode]


for var in gen_var_names:
    print(var)
    binning = opt.dedicated_binning[var]    
    mask = mc_info[data_name].fiducial_masks==1

    data_var = mc_info[data_name].file[var][:flags.N][mask]
    if 'tau' in var:
        data_var = np.log(data_var)
    fig,gs = opt.SetGrid() 
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1],sharex=ax0)
    xaxis = [(binning[i] + binning[i+1])/2.0 for i in range(len(binning)-1)]
    if 'genjet_pt' in var or 'Q2' in var:
        ax0.set_xscale('log')
        ax0.set_yscale('log')
        ax1.set_xscale('log')


        
    data_pred,_,_=ax0.hist(data_var,weights=weight_data[mask]*mc_info[data_name].nominal_wgts[mask],bins=binning,label="Data",density=True,color="black",histtype="step")
    opt.FormatFig(xlabel = "", ylabel = r'$1/\sigma$ $\mathrm{d}\sigma/\mathrm{d}$%s'%gen_var_names[var],ax0=ax0)

    ax0.tick_params(axis='x',labelsize=0)
    ratios = {}
    
    for mc_name in mc_names:
        #Upper canvas
        mc = mc_name.split("_")[0]
        mask = mc_info[mc_name].fiducial_masks==1
        mc_var = mc_info[mc_name].file[var][:flags.N][mask]

        if 'tau' in var:
            mc_var = np.log(mc_var)    
        
        pred,_,_=ax0.hist(mc_var,weights=mc_info[mc_name].nominal_wgts[mask],bins=binning,label=mc,density=True,color=opt.colors[mc],histtype="step")        
        ratios[mc] = 100*np.divide(pred-data_pred,data_pred)        
        #Ratio plot    
        mc = mc_name.split("_")[0]
        ax1.plot(xaxis,ratios[mc],color=opt.colors[mc],marker=opt.markers[mc],ms=12,lw=0,markerfacecolor='none',markeredgewidth=3)


    ax0.legend(loc='lower right',fontsize=16,ncol=1)    
    RatioLabel(ax1)
    
    plot_folder = '../plots_'+data_name if flags.closure==False else '../plots_closure'    
    plot_folder+='_{}'.format(flags.mode)
    
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
    fig.savefig(os.path.join(plot_folder,"{}.pdf".format(var)))


    

    if flags.comp:
        #Compare the non-closure uncertainty
        fig,gs = opt.SetGrid() 
        ax0 = plt.subplot(gs[0])
        ax0.tick_params(axis='x',labelsize=0)
        
        ax1 = plt.subplot(gs[1],sharex=ax0)

        if 'genjet_pt' in var or 'Q2' in var:
            ax0.set_xscale('log')
            ax0.set_yscale('log')
            ax1.set_xscale('log')

        
        mask = mc_info[mc_ref].fiducial_masks==1        
        mc_var = mc_info[mc_ref].file[var][:flags.N][mask]
        if 'tau' in var:
            mc_var = np.log(mc_var)    

        mc_pred,_,_=ax0.hist(mc_var,weights=mc_info[mc_ref].nominal_wgts[mask],bins=binning,label="Target Gen",density=True,color="black",histtype="step")

        opt.FormatFig(xlabel = "", ylabel = r'$1/\sigma$ $\mathrm{d}\sigma/\mathrm{d}$%s'%gen_var_names[var],ax0=ax0)

        ratios = {}        
        mask = mc_info[data_name].fiducial_masks==1
        data_var = mc_info[data_name].file[var][:flags.N][mask]
        if 'tau' in var:
            data_var = np.log(data_var)    
        for train in weights_data:
            pred,_,_ = ax0.hist(data_var,weights=weights_data[train][mask]*mc_info[data_name].nominal_wgts[mask],bins=binning,label=train,density=True,color=opt.colors[train],histtype="step")
            ratios[train] = 100*np.divide(mc_pred-pred,pred)
            
            ax1.plot(xaxis,ratios[train],color=opt.colors[train],marker=opt.markers[train],ms=12,lw=0,markerfacecolor='none',markeredgewidth=3)

            
        RatioLabel(ax1)
        ax0.legend(loc='lower right',fontsize=16,ncol=1)
        
        plot_folder = '../plots_comp'
        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)
        fig.savefig(os.path.join(plot_folder,"{}_{}.pdf".format(var,flags.niter)))






