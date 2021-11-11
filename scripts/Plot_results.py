import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LogNorm
from sklearn.preprocessing import StandardScaler
from matplotlib import gridspec
from matplotlib.font_manager import FontProperties
import argparse
import os
import tensorflow as tf
import tensorflow.keras
import tensorflow.keras.backend as K
from pct import PCT
import options as opt
import h5py as h5
from unfold_hvd import  Multifold


opt.SetStyle()

parser = argparse.ArgumentParser()

#parser.add_argument('--data_folder', default='/global/cfs/cdirs/m1759/vmikuni/H1/pkl', help='Folder containing data and MC files')
parser.add_argument('--data_folder', default='/pscratch/sd/v/vmikuni', help='Folder containing data and MC files')
parser.add_argument('--closure', action='store_true', default=False,help='Plot closure results')
parser.add_argument('--pct', action='store_true', default=False,help='Load pct results')
parser.add_argument('--comp', action='store_true', default=False,help='Compare the non-closure unc between methods')
parser.add_argument('-N', type=int,default=5e6, help='Number of events to evaluate')
parser.add_argument('--niter', type=int, default=9, help='Omnifold iteration to load')

flags = parser.parse_args()
flags.N = int(flags.N)

mc_names = ['Rapgap','Djangoh']
mc_tags = ['nominal','nominal']
    
data_idx = 0
data_name = mc_names[data_idx]
data_tag = mc_tags[data_idx]
folder = 'results'

if flags.closure:
    version = 'closure'
else:
    version = 'nominal'


gen_var_names = {
    'genjet_pt': r"$p_\mathrm{T}^\mathrm{jet}$",
    'genjet_eta':r'$\eta^\mathrm{jet}$',
    'gen_Q2':r'$Q^2$',


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


if flags.pct:
    var_names = ['gen_jet_part_eta','gen_jet_part_phi','gen_jet_part_pt','gen_jet_part_charge']
else:
    var_names = ['genjet_pt','genjet_eta','genjet_phi','gen_jet_ncharged','gen_Q2',
                 'gen_jet_charge', 'gen_jet_ptD','gen_jet_tau10', 'gen_jet_tau15', 'gen_jet_tau20']
    
    

predictions = {}
nominal_wgts = {}
fiducial_masks = {}
truth_mask = {}
reco_mask = {}

for mc,tag in zip(mc_names,mc_tags):
    print("{}_{}.h5".format(mc,tag))
    predictions[mc] = h5.File(os.path.join(flags.data_folder,"{}_{}.h5".format(mc,tag)),'r')
    nominal_wgts[mc] = predictions[mc]['wgt'][:flags.N]
    fiducial_masks[mc] = predictions[mc]['pass_fiducial'][:flags.N] #pass fiducial region definition
    truth_mask[mc] = predictions[mc]['pass_truth'][:flags.N] #pass truth region definition
    #fiducial_masks[mc] *= predictions[mc]['pass_reco'][:flags.N]
    reco_mask[mc] = predictions[mc]['pass_reco'][:flags.N]
    if mc == data_name:
        base_name = "Omnifold"
        if flags.pct:
            base_name+='_PCT'
            
        model_name = '../weights/{}_{}_iter{}_step2.h5'.format(base_name,version,flags.niter)
        data = np.concatenate([np.expand_dims(predictions[mc][var][:flags.N],-1) for var in var_names],-1)
        if flags.pct:
            data[:,0,0][truth_mask[mc]==0] = -10
        else:
            data[:,0][truth_mask[mc]==0] = -10
        # print(np.average(predictions[mc]['gen_jet_ncharged'][:flags.N]/np.sum(data[:,:,2]>0,1)))
        # input()
        Q2 = np.ma.log(predictions[mc]['gen_Q2'][:flags.N]).filled(-1)
        if flags.pct:            
            jet_pt = predictions[mc]['genjet_pt'][:flags.N]/100
            #data[:,:,2]*=np.expand_dims(jet_pt,-1)
            data = [data,Q2]
            
        mfold = Multifold(
            nvars=len(var_names),
            niter=1,
            pct=flags.pct,
            Q2=Q2,
        )
        
        mfold.mc_gen = data
        if flags.pct == False:
            scaler = StandardScaler()
            scaler.fit(mfold.mc_gen)
            mfold.mc_gen=scaler.transform(mfold.mc_gen)


        mfold.PrepareModel()
        mfold.model.load_weights(model_name)
        weights_data = mfold.reweight(mfold.mc_gen)
        print(weights_data)
        del mfold
        del data
        
        
for var in gen_var_names:
    print(var)
    #fig,ax = plt.subplots(figsize=(9,7))
    fig = plt.figure(figsize=(9, 9))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3,1]) 
    gs.update(wspace=0.025, hspace=0.1)
 
    mask = fiducial_masks[data_name]==1

    ax0 = plt.subplot(gs[0])
    binning = opt.dedicated_binning[var]
    to_plot = predictions[data_name][var][:flags.N][mask]
    if 'tau' in var:
        to_plot = np.log(to_plot)    
    data_pred,_,_=ax0.hist(to_plot,weights=weights_data[mask]*nominal_wgts[data_name][mask],bins=binning,label="Data",density=True,color="black",histtype="step")
    
    plt.xlabel(gen_var_names[var])
    plt.ylabel(r'$1/\sigma$ $\mathrm{d}\sigma/\mathrm{d}$%s'%gen_var_names[var])
    if 'jet_pt' in var:
         plt.ylabel(r'$1/\sigma$ $\mathrm{d}\sigma/\mathrm{d}$%s [1/GeV]'%gen_var_names[var])
    
    xposition = 0.8
    plt.text(xposition, 0.92,'H1 Internal',
             horizontalalignment='center',
             verticalalignment='center',
             transform = ax0.transAxes, fontsize=25, fontweight='bold')
    plt.text(xposition, 0.82,'Iteration {}'.format(flags.niter),
             horizontalalignment='center',
             verticalalignment='center',
             transform = ax0.transAxes, fontsize=25, fontweight='bold')
    ratios = {}
    for mc in mc_names:
        mask = fiducial_masks[mc]==1
        to_plot = predictions[mc][var][:flags.N][mask]
        if 'tau' in var:
            to_plot = np.log(to_plot)    
        
        pred,_,_=ax0.hist(to_plot,weights=nominal_wgts[mc][mask],bins=binning,label=mc,density=True,color=opt.colors[mc],histtype="step")
        ratios[mc] = 100*np.divide(pred-data_pred,data_pred)
        
    ax0.legend(loc='lower right',fontsize=16,ncol=1)
    #plt.show()
    plt.xlabel("",fontsize=20)
    plt.xticks(fontsize=0)
    if 'genjet_pt' in var or 'Q2' in var:
        plt.xscale('log')
        plt.yscale('log')

    ax1 = plt.subplot(gs[1],sharex=ax0)
    xaxis = [(binning[i] + binning[i+1])/2.0 for i in range(len(binning)-1)]
    for mc in mc_names:
        ax1.plot(xaxis,ratios[mc],color=opt.colors[mc],marker=opt.markers[mc],ms=12,lw=0,markerfacecolor='none',markeredgewidth=3)
        
    plt.ylabel('Model unc. (%)')
    plt.xlabel(gen_var_names[var])
    plt.axhline(y=0.0, color='r', linestyle='-')
    plt.axhline(y=10, color='r', linestyle='--')
    plt.axhline(y=-10, color='r', linestyle='--')

    
    if 'genjet_pt' in var:
        plt.xscale('log')


    plt.ylim([-20,20])
    #plt.tight_layout()
    
    plot_folder = '../plots_'+data_name if flags.closure==False else 'plots_closure'
    if flags.pct:
        plot_folder+='_pct'
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
    fig.savefig(os.path.join(plot_folder,"{}_{}.png".format(var,flags.niter)))
