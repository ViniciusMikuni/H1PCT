import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LogNorm
from sklearn.preprocessing import StandardScaler
from matplotlib import gridspec
from matplotlib.font_manager import FontProperties
from numpy import inf
import argparse
import os
import tensorflow as tf
import tensorflow.keras
import tensorflow.keras.backend as K
from pct import PCT
import options as opt
import h5py as h5
from unfold import  Multifold


opt.SetStyle()

parser = argparse.ArgumentParser()

parser.add_argument('--data_folder', default='/clusterfs/ml4hep/vmikuni/H1/jet_subs/h5', help='Folder containing data and MC files')
#parser.add_argument('--data_folder', default='/pscratch/sd/v/vmikuni', help='Folder containing data and MC files')
parser.add_argument('--closure', action='store_true', default=False,help='Plot closure results')
parser.add_argument('--comp', action='store_true', default=False,help='Compare closure unc. from different methods')
parser.add_argument('--pct', action='store_true', default=False,help='Load pct results')
parser.add_argument('-N', type=int,default=5e6, help='Number of events to evaluate')
parser.add_argument('--niter', type=int, default=9, help='Omnifold iteration to load')

flags = parser.parse_args()
flags.N = int(flags.N)

mc_names = ['Rapgap','Djangoh']
mc_tags = ['nominal','nominal']
    
data_idx = 0
data_name = mc_names[data_idx]
data_tag = mc_tags[data_idx]
mc_ref = mc_names[data_idx-1]
print(mc_ref)
folder = 'results'
if flags.closure:
    version = 'closure'
else:
    version = 'nominal'


gen_var_names = {
    'gen_Q2': r"$Q^2$",
    # 'genjet_pt': r"$p_\mathrm{T}^\mathrm{jet}$",
    # 'genjet_eta':r'$\eta^\mathrm{jet}$',


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


var_names = ['genjet_pt','genjet_eta','genjet_phi','gen_jet_ncharged','gen_Q2',
             'gen_jet_charge', 'gen_jet_ptD','gen_jet_tau10', 'gen_jet_tau15', 'gen_jet_tau20']
    
    
class MCInfo():
    def __init__(self,mc,tag,N,data_folder):
        self.mc = mc
        self.tag = tag
        self.N = N
        self.predictions = h5.File(os.path.join(data_folder,"{}_{}.h5".format(mc,tag)),'r')
        self.nominal_wgts = self.predictions['wgt'][:self.N]
        self.fiducial_masks = self.predictions['pass_fiducial'][:self.N] #pass fiducial region definition
        self.truth_mask = self.predictions['pass_truth'][:self.N] #pass truth region definition
        

    def LoadDataWeights(self,niter,pct=False):
        base_name = "Omnifold"
        if pct:
            base_name+='_PCT'            
        model_name = 'weights/{}_{}_iter{}_step2.h5'.format(base_name,version,niter)

        data = np.concatenate([np.expand_dims(self.predictions[var][:self.N],-1) for var in var_names],-1)

        data[:,0][self.truth_mask==0] = -10
                    
        mfold = Multifold(
            nvars=len(var_names),
            niter=1,
            pct=pct,
            Q2=None,
        )

        scaler = StandardScaler()
        scaler.fit(data)
        data=scaler.transform(data)


        mfold.mc_gen = data
        mfold.PrepareModel()
        mfold.model2.load_weights(model_name)
        return mfold.reweight(mfold.mc_gen,mfold.model2)


mc_info = {}
weights_data = {}

for mc,tag in zip(mc_names,mc_tags):
    print("{}_{}.h5".format(mc,tag))
    mc_info[mc] = MCInfo(mc,tag,flags.N,flags.data_folder)
    
    if mc == data_name:
        weights_data['MLP'] = mc_info[mc].LoadDataWeights(flags.niter,pct=False)
        weights_data['PCT'] = mc_info[mc].LoadDataWeights(flags.niter,pct=True)
        
if flags.pct:
    weight_data = weights_data['PCT']
else:
    weight_data = weights_data['MLP']
        

def SetGrid():
    fig = plt.figure(figsize=(9, 9))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3,1]) 
    gs.update(wspace=0.025, hspace=0.1)
    return fig,gs

def FormatFig(xlabel,ylabel):
    y_loc, _ = plt.yticks()
    y_update = ['%.2f' % y for y in y_loc]
    plt.yticks(y_loc, y_update)    
    plt.xlabel(xlabel,fontsize=20)
    plt.ylabel(ylabel)
        

    xposition = 0.8
    plt.text(xposition, 0.92,'H1 Internal',
             horizontalalignment='center',
             verticalalignment='center',
             transform = ax0.transAxes, fontsize=25, fontweight='bold')



for var in gen_var_names:
    print(var)
    binning = opt.dedicated_binning[var]    
    mask = mc_info[data_name].fiducial_masks==1

    data_var = mc_info[data_name].predictions[var][:flags.N][mask]
    if 'tau' in var:
        data_var = np.log(data_var)    

    fig,gs = SetGrid() 
    ax0 = plt.subplot(gs[0])
    

    data_pred,_,_=ax0.hist(data_var,weights=weight_data[mask]*mc_info[data_name].nominal_wgts[mask],bins=binning,label="Data",density=True,color="black",histtype="step")

    FormatFig(xlabel = "", ylabel = r'$1/\sigma$ $\mathrm{d}\sigma/\mathrm{d}$%s'%gen_var_names[var])
    
    ratios = {}
    for mc in mc_names:
        mask = mc_info[mc].fiducial_masks==1
        mc_var = mc_info[mc].predictions[var][:flags.N][mask]
        if 'tau' in var:
            mc_var = np.log(mc_var)    
        
        pred,_,_=ax0.hist(mc_var,weights=mc_info[mc].nominal_wgts[mask],bins=binning,label=mc,density=True,color=opt.colors[mc],histtype="step")
        ratios[mc] = 100*np.divide(pred-data_pred,data_pred)
        
    ax0.legend(loc='lower right',fontsize=16,ncol=1)
    plt.xticks(fontsize=0)


    ax1 = plt.subplot(gs[1],sharex=ax0)
    xaxis = [(binning[i] + binning[i+1])/2.0 for i in range(len(binning)-1)]


    for mc in mc_names:
        ax1.plot(xaxis,ratios[mc],color=opt.colors[mc],marker=opt.markers[mc],ms=12,lw=0,markerfacecolor='none',markeredgewidth=3)
        
    plt.ylabel('Model unc. (%)')
    plt.xlabel(gen_var_names[var])
    plt.axhline(y=0.0, color='r', linestyle='-')
    plt.axhline(y=10, color='r', linestyle='--')
    plt.axhline(y=-10, color='r', linestyle='--')

    plt.ylim([-20,20])
    
    plot_folder = 'plots_'+data_name if flags.closure==False else 'plots_closure'
    if flags.pct:
        plot_folder+='_pct'
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
    fig.savefig(os.path.join(plot_folder,"{}.pdf".format(var)))

    if flags.comp:
        #Compare the non-closure uncertainty
        fig,gs = SetGrid() 
        ax0 = plt.subplot(gs[0])
        

        mc_var = mc_info[mc_ref].predictions[var][:flags.N][mask]
        if 'tau' in var:
            mc_var = np.log(mc_var)    

        mask = mc_info[mc_ref].fiducial_masks==1
        mc_pred,_,_=ax0.hist(mc_var,weights=mc_info[mc_ref].nominal_wgts[mask],bins=binning,label="Target Gen",density=True,color="black",histtype="step")

        FormatFig(xlabel = "", ylabel = r'$1/\sigma$ $\mathrm{d}\sigma/\mathrm{d}$%s'%gen_var_names[var])

        ratios = {}        
        mask = mc_info[data_name].fiducial_masks==1
        data_var = mc_info[data_name].predictions[var][:flags.N][mask]
        if 'tau' in var:
            data_var = np.log(data_var)    
        for train in weights_data:
            pred,_,_ = ax0.hist(data_var,weights=weights_data[train][mask]*mc_info[data_name].nominal_wgts[mask],bins=binning,label=train,density=True,color=opt.colors[train],histtype="step")
            ratios[train] = 100*np.divide(mc_pred-pred,pred)


        ax0.legend(loc='lower right',fontsize=16,ncol=1)
        plt.xticks(fontsize=0)


        ax1 = plt.subplot(gs[1],sharex=ax0)

        for train in weights_data:
            ax1.plot(xaxis,ratios[train],color=opt.colors[train],marker=opt.markers[train],ms=12,lw=0,markerfacecolor='none',markeredgewidth=3)
        
        plt.ylabel('Model unc. (%)')
        plt.xlabel(gen_var_names[var])
        plt.axhline(y=0.0, color='r', linestyle='-')
        plt.axhline(y=10, color='r', linestyle='--')
        plt.axhline(y=-10, color='r', linestyle='--')

        plt.ylim([-20,20])
    
        plot_folder = 'plots_comp'
        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)
        fig.savefig(os.path.join(plot_folder,"{}.pdf".format(var)))



