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
import options as opt
import h5py as h5


opt.SetStyle()

parser = argparse.ArgumentParser()

parser.add_argument('--data_folder', default='/clusterfs/ml4hep/vmikuni/H1/jet_subs/h5', help='Folder containing data and MC files')
parser.add_argument('-N', type=int,default=5e6, help='Number of events to evaluate')


flags = parser.parse_args()
flags.N = int(flags.N)

mc_names = ['Rapgap','Djangoh']
mc_tags = ['nominal','nominal']
    

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




class MCInfo():
    def __init__(self,mc,tag,N,data_folder):
        self.N = N
        self.predictions = h5.File(os.path.join(data_folder,"{}_{}.h5".formt(mc,tag)),'r')
        self.nominal_wgts = self.predictions['wgt'][:self.N]
        self.fiducial_masks = self.predictions['pass_reco'][:self.N] #pass fiducial region definition

mc_info = {}

for mc,tag in zip(mc_names,mc_tags):
    print("{}_{}.h5".format(mc,tag))
    mc_info[mc] = MCInfo(mc,tag,flags.N,flags.data_folder)


for var in var_names:
    print(var)
    binning = opt.dedicated_binning[var]    

    data_var = data_vars[var][:]

    if 'tau' in var:
        data_var = np.log(data_var)    

    fig,gs = opt.SetGrid() 
    ax0 = plt.subplot(gs[0])
    

    data_pred,_,_=ax0.hist(data_var,bins=binning,label="Data",density=True,color="black",histtype="step")
    print(var,data_pred)

    opt.FormatFig(xlabel = "", ylabel = 'Normalized Events / bin',ax0=ax0)
    if var == 'jet_pt':
        plt.yscale("log")  
        plt.xscale("log")  
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
        
    plt.ylabel('Difference. (%)')
    plt.xlabel(var_names[var])
    plt.axhline(y=0.0, color='r', linestyle='-')
    plt.axhline(y=10, color='r', linestyle='--')
    plt.axhline(y=-10, color='r', linestyle='--')

    plt.ylim([-20,20])
    
    plot_folder = '../plots_data'
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
    fig.savefig(os.path.join(plot_folder,"{}.pdf".format(var)))
