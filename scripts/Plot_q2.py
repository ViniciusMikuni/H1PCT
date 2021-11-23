import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LogNorm
from matplotlib import gridspec
from matplotlib.font_manager import FontProperties
from numpy import inf
import argparse
import os
import h5py as h5
import sys
sys.path.append('../')
import shared.options as opt


opt.SetStyle()

parser = argparse.ArgumentParser()

parser.add_argument('--data_folder', default='/clusterfs/ml4hep/vmikuni/H1/jet_subs/h5', help='Folder containing data and MC files')


flags = parser.parse_args()

    

data_name = 'data'
data_vars = h5.File(os.path.join(flags.data_folder,"{}.h5".format(data_name)),'r')



var_names = {
    'Q2': r"$Q^2$",
    'jet_pt': r"$p_\mathrm{T}^\mathrm{jet}$",
    'jet_eta':r'$\eta^\mathrm{jet}$',


    # 'genjet_phi':r'$\phi^\mathrm{jet}$',
    'jet_ncharged':r'$\mathrm{N_{part}}^\mathrm{jet}$', 
    'jet_charge':r'$\mathrm{Q}^\mathrm{jet}$', 
    'jet_ptD':r'$p_\mathrm{T}\mathrm{D}^\mathrm{jet}$',
    'jet_tau10':r'$\mathrm{log}(\tau_{1}^\mathrm{jet})$', 
    'jet_tau15':r'$\mathrm{log}(\tau_{0.5}^\mathrm{jet})$',
    'jet_tau20':r'$\mathrm{log}(\tau_{0}^\mathrm{jet})$',

    # 'gene_px':'e px',
    # 'gene_py':'e px',
    # 'gene_pz':'e px',
}

q2_bins = opt.dedicated_binning['Q2']

for var in var_names:
    print(var)
    binning = opt.dedicated_binning[var]    

    data_var = data_vars[var][:]

    if 'tau' in var:
        data_var = np.log(data_var)    

    fig,gs = opt.SetGrid(ratio=False) 
    ax0 = plt.subplot(gs[0])
    opt.FormatFig(xlabel = var_names[var], ylabel = 'Normalized Events / bin',ax0=ax0)
    cmap= plt.get_cmap('PiYG')
    cmap= plt.get_cmap('cool')
    for iq,q2 in enumerate(q2_bins):
        if iq==len(q2_bins)-1:continue
        mask = (data_vars['Q2'][:] > q2) & (data_vars['Q2'][:] < q2_bins[iq+1])

        data_pred,_, = np.histogram(data_var[mask],bins=binning)
        norm = np.sum(data_pred)
        data_pred=data_pred/norm
        xaxis = [(binning[i] + binning[i+1])/2.0 for i in range(len(binning)-1)]
        plt.errorbar(
            xaxis,
            data_pred,
            yerr = (data_pred/norm)**0.5,
            marker = '.',
            drawstyle = 'steps-mid',            
            label="{} < Q2 < {} GeV".format(int(np.round(q2,1)),int(np.round(q2_bins[iq+1],1))),
        )
        #data_pred,_,_=ax0.hist(data_var[mask],bins=binning,label="{} < Q2 < {} GeV".format(np.round(q2,1),np.round(q2_bins[iq+1],1)),density=True,color=cmap(1.0*iq/len(q2_bins)),histtype="step")

        if var == 'jet_pt' or var == 'Q2':
            plt.yscale("log")  
            plt.xscale("log")  

        
    ax0.legend(loc='best',fontsize=16,ncol=1)

    
    plot_folder = '../plots_q2'
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
    fig.savefig(os.path.join(plot_folder,"{}.pdf".format(var)))
