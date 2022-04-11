import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.font_manager import FontProperties
import argparse
import os
import h5py as h5
from omnifold import  Multifold, Scaler, LoadJson
from SaveWeights import MCInfo
import sys
sys.path.append('../')
import shared.options as opt


opt.SetStyle()

parser = argparse.ArgumentParser()

parser.add_argument('--data_folder', default='/pscratch/sd/v/vmikuni/H1', help='Folder containing data and MC files')
parser.add_argument('--weights', default='../weights', help='Folder to store trained weights')
parser.add_argument('--mode', default='hybrid', help='Which train type to load [hybrid/standard/PCT]')
parser.add_argument('--config', default='config_general.json', help='Basic config file containing general options')


parser.add_argument('--sys', action='store_true', default=False,help='Evaluate results with systematic uncertainties')
parser.add_argument('--plot_reco', action='store_true', default=False,help='Plot reco level comparison between data and MC predictions')
parser.add_argument('-N',type=float,default=20e6, help='Number of events to evaluate')
parser.add_argument('--niter', type=int, default=6, help='Omnifold iteration to load')
parser.add_argument('--img_fmt', default='pdf', help='Format of the output figures')

flags = parser.parse_args()
flags.N = int(flags.N)

config=LoadJson(flags.config)


mc_names = ['Rapgap_nominal','Djangoh_nominal']
#standalone_predictions = ['Herwig','Pythia','Pythia_Vincia','Pythia_Dire']
#'Herwig_Matchbox'
standalone_predictions = []    
data_idx = 0 #Sample that after weights represent data
data_name = mc_names[data_idx]

mc_ref = mc_names[data_idx-1] #MC ref is used to define the reference simulation used to derive the closure and model systematic uncertainties
print(mc_ref)

text_ypos = 0.7 #text position height
text_xpos = 0.22
version = data_name

if flags.plot_reco:
    gen_var_names = opt.reco_vars
else:
    gen_var_names = opt.gen_vars


def RatioLabel(ax1,var):
    ax1.set_ylabel('Rel. diff. [%]')
    ax1.set_xlabel(r'$Q^2$ [GeV]')    
    ax1.axhline(y=0.0, color='r', linestyle='-')
    # ax1.axhline(y=10, color='r', linestyle='--')
    # ax1.axhline(y=-10, color='r', linestyle='--')
    ylim = [-20,20]
    ax1.set_ylim(ylim)
    


def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return (average, np.sqrt(variance))


def LoadData(q2_int):
    mc_info = {}
    weights_data = {}
    sys_variations = {}
    mc_info['data'] = MCInfo('data',flags.N,flags.data_folder,config,q2_int,is_reco=True)  
    #Loading weights from training
    for mc_name in mc_names:
        print("{}.h5".format(mc_name))    
        mc_info[mc_name] = MCInfo(mc_name,flags.N,flags.data_folder,config,q2_int,is_reco=flags.plot_reco)
        if mc_name == data_name:
            base_name = "Omnifold_{}".format(flags.mode)
            model_name = '{}/{}_{}_iter{}_step2.h5'.format(flags.weights,base_name,version,flags.niter)


            weights_data[flags.mode] = mc_info[mc_name].ReturnWeights(flags.niter,model_name=model_name,mode=flags.mode)


            if flags.sys == True: #load systematic variations
                for unc in opt.sys_sources:
                    if unc in ['model','closure','stat','QED']: continue
                    model_name = '{}/{}_{}_iter{}_step2.h5'.format(
                        flags.weights,base_name,version.replace("nominal",unc),flags.niter)
                    print(mc_name.replace("nominal",unc))
                    mc_info[unc] = MCInfo(mc_name.replace("nominal",unc),int(flags.N),flags.data_folder,config,q2_int,is_reco=flags.plot_reco)
                    sys_variations[unc] = mc_info[unc].ReturnWeights(
                        flags.niter,model_name=model_name,mode=flags.mode)
                    
                if not flags.plot_reco:
                    #Load non-closure weights
                    model_name = '{}/{}_{}_iter{}_step2.h5'.format(
                        flags.weights,base_name,version+'_closure',flags.niter)
                    sys_variations['closure'] = mc_info[mc_name].ReturnWeights(
                        flags.niter,model_name=model_name,mode=flags.mode)
                
                    sys_variations['stat'] = [
                        mc_info[mc_name].LoadTrainedWeights(
                            os.path.join(flags.data_folder,'weights','{}_{}.h5'.format(mc_name,nstrap))
                        ) for nstrap in range(1,config['NBOOTSTRAP']+1)]
                else:
                    sys_variations['stat'] = []
                    
        elif flags.sys and not flags.plot_reco:
            base_name = "Omnifold_{}".format(flags.mode)
            model_name = '{}/{}_{}_iter{}_step2.h5'.format(flags.weights,base_name,mc_name,flags.niter)
            sys_variations['model'] = mc_info[mc_name].ReturnWeights(
                flags.niter,model_name=model_name,mode=flags.mode)
            

    weight_data = weights_data[flags.mode]
    return mc_info,weight_data,sys_variations



binning = opt.dedicated_binning['Q2']
for var in gen_var_names:
    print(var)
    fig,gs = opt.SetGrid(1) 
    ax0 = plt.subplot(gs[0])
    #ax0.tick_params(axis='x',labelsize=0)
    # ax1 = plt.subplot(gs[1],sharex=ax0)
    ax0.set_xscale('log')
    opt.FormatFig(xlabel = r'$Q^2$ [GeV]', ylabel = r'<%s>'%gen_var_names[var],ax0=ax0)
    # ax1.set_xscale('log')

    xaxis = 0.5*(binning[:-1] + binning[1:])

    data_pred = []
    sys_unc = []
    stat_unc = []
    mc_pred = {}
    for mc_name in mc_names:
        mc_pred[mc_name] = []

        
    for q2_int in range(1,5):  
        mc_info,weight_data,sys_variations = LoadData(q2_int)
        if flags.plot_reco:
            data_var = mc_info['data'].LoadVar(var)
            data_mean = np.average(data_var)
        else:
            data_var = mc_info[data_name].LoadVar(var)
            if '_charge' in var:
                mask_var = np.abs(data_var)<0.9
            else:
                mask_var = np.abs(data_var)>=0
            data_mean,data_std = weighted_avg_and_std(data_var[mask_var],weights=(weight_data*mc_info[data_name].nominal_wgts)[mask_var])
        data_pred.append(data_mean)
        #######################################
        # Processing systematic uncertainties #
        #######################################
        if flags.sys == True:
            if not flags.plot_reco:
                #Stat uncertainty
                straps = []
                for strap in range(len(sys_variations['stat'])):
                    sys_pred,sts_std = weighted_avg_and_std(data_var,weights=sys_variations['stat'][strap]*mc_info[data_name].nominal_wgts)             
                    straps.append(sys_pred)
                stat_unc.append(100*np.std(straps,axis=0)/np.mean(straps,axis=0))
            else:
                #FixMe: correct this thing
                stat_unc.append(0)
                
            total_sys=0  
            for unc in opt.sys_sources:
                if unc == 'stat': continue
                if unc == 'QED': continue
                if unc == 'closure':continue
                elif unc == 'model':
                    #Model uncertainty: difference between unfolded values
                    data_sys = mc_info[mc_ref].LoadVar(var)
                    if '_charge' in var:
                        mask_var = np.abs(data_sys)<0.9
                    else:
                        mask_var = np.abs(data_sys)>=0
                    sys_pred,sys_std = weighted_avg_and_std(data_sys[mask_var],weights=(sys_variations[unc]*mc_info[mc_ref].nominal_wgts)[mask_var])
                    ratio_sys = 100*np.divide(sys_pred-data_mean,data_mean)
                    #ratio_sys= np.sqrt(np.abs(ratio_sys**2 - stat_unc[-1]**2))
                else:
                    data_sys = mc_info[unc].LoadVar(var)
                    if '_charge' in var:
                        mask_var = np.abs(data_sys)<0.9
                    else:
                        mask_var = np.abs(data_sys)>=0
                    if flags.plot_reco:
                        sys_pred = np.average(data_sys,weights=mc_info[unc].nominal_wgts)
                        mc_var = mc_info[data_name].LoadVar(var)
                        pred,_=np.average(mc_var,weights=mc_info[data_name].nominal_wgts)
                        ratio_sys = 100*np.divide(sys_pred-pred,pred)
                    else:
                        sys_pred,sys_std = weighted_avg_and_std(data_sys[mask_var],weights=(sys_variations[unc]*mc_info[unc].nominal_wgts)[mask_var])
                        ratio_sys = 100*np.divide(sys_pred-data_mean,data_mean)
                        
                
                total_sys+= ratio_sys**2
                print(ratio_sys**2,unc)
            #total_sys = np.abs(total_sys - stat_unc[-1]**2)
            print(total_sys)
            sys_unc.append(total_sys)
        else:
            sys_unc.append(0)
            stat_unc.append(0)

        ################################
        # Processing other predictions #
        ################################

        for mc_name in mc_names:
            #Upper canvas
            mc_var = mc_info[mc_name].LoadVar(var)
            if '_charge' in var:
                mask_var = np.abs(mc_var)<0.9
            else:
                mask_var = np.abs(mc_var)>=0
            
            mc_pred[mc_name].append(np.average(mc_var[mask_var],weights=mc_info[mc_name].nominal_wgts[mask_var]))

    #Lets plot the results
    data_pred = np.array(data_pred)
    #ax0.set_ylim(top=2.5*max(data_pred),bottom = 0.5*min(data_pred))
    ax0.set_xlim(left=95,right=5.5e3)
    if max(data_pred) <0:
        ax0.set_ylim(top=0.5*min(data_pred),bottom = 1.3*max(data_pred))
    else:
        ax0.set_ylim(top=1.3*max(data_pred),bottom = 0.7*min(data_pred))
        #Axis is reversed for some reason
        #ax0.invert_yaxis()

        
    plt.text(text_xpos, text_ypos,'$Q^{2}>$150 GeV$^{2}$ \n $0.2<y<0.7$ \n $p_\mathrm{T}^\mathrm{jet}>10$ GeV  \n $k_\mathrm{T}, R=1.0$',
             horizontalalignment='center',
             verticalalignment='center',
             transform = ax0.transAxes, fontsize=18)
        
    ax0.errorbar(xaxis,data_pred,yerr = np.abs(data_pred)*np.sqrt(sys_unc)/100.0,fmt='o',ms=12,color='k',label='Data')
    for ibin in range(len(xaxis)):
        xup = binning[ibin+1]
        xlow = binning[ibin]
        #ax1.fill_between(np.array([xlow,xup]),np.sqrt(sys_unc[ibin]),-np.sqrt(sys_unc[ibin]), alpha=0.3,color='k')

    
    for mc_name in mc_names:
        mc = mc_name.split("_")[0]
        ax0.plot(xaxis,mc_pred[mc_name],color=opt.colors[mc],marker=opt.markers[mc],ms=12,lw=0,markerfacecolor='none',markeredgewidth=3,label=mc)
        ratio = 100*np.divide(np.array(mc_pred[mc_name])-data_pred,data_pred)
        #ax1.plot(xaxis,ratio,color=opt.colors[mc],marker=opt.markers[mc],ms=12,lw=0,markerfacecolor='none',markeredgewidth=3)
        
    ax0.legend(loc='upper left',fontsize=16,ncol=2)    
    #RatioLabel(ax1,var)
            
    plot_folder = '../plots_average_'+data_name
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
    fig.savefig(os.path.join(plot_folder,"{}_{}.{}".format(var,flags.niter,flags.img_fmt)))
