import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import argparse
import os
import h5py as h5
from omnifold import  Multifold, Scaler, LoadJson
from SaveWeights import MCInfo
import sys
sys.path.append('../')
import shared.options as opt

from matplotlib.ticker import StrMethodFormatter


opt.SetStyle()
parser = argparse.ArgumentParser()

# parser.add_argument('--data_folder', default='/pscratch/sd/v/vmikuni/H1', help='Folder containing data and MC files')
parser.add_argument('--data_folder', default='/global/cfs/cdirs/m3929/SCRATCH/H1/', help='Folder containing data and MC files')
parser.add_argument('--weights', default='../weights', help='Folder to store trained weights')
parser.add_argument('--mode', default='hybrid', help='Which train type to load [hybrid/standard/PCT]')
parser.add_argument('--config', default='config_general.json', help='Basic config file containing general options')


parser.add_argument('--sys', action='store_true', default=False,help='Evaluate results with systematic uncertainties')
parser.add_argument('--mom', default=1, help='Data moment to calculate')
parser.add_argument('--plot_reco', action='store_true', default=False,help='Plot reco level comparison between data and MC predictions')
parser.add_argument('-N',type=float,default=20e6, help='Number of events to evaluate')
parser.add_argument('--niter', type=int, default=6, help='Omnifold iteration to load')
parser.add_argument('--img_fmt', default='pdf', help='Format of the output figures')

flags = parser.parse_args()
flags.N = int(flags.N)

config=LoadJson(flags.config)
add_predictions = LoadJson('jet_substructure_moments.json')


# if flags.plot_reco:
#     raise ValueError('Plots at reco level are not yet fully implemented!')

mc_names = ['Rapgap_nominal','Djangoh_nominal']
standalone_predictions = []    
data_idx = 0 #Sample that after weights represent data
data_name = mc_names[data_idx]

mc_ref = mc_names[data_idx-1] #MC ref is used to define the reference simulation used to derive the closure and model systematic uncertainties
print(mc_ref)

text_ypos = 0.8 #text position height
text_xpos = 0.8
version = data_name

if flags.plot_reco:
    gen_var_names = opt.reco_vars
else:
    gen_var_names = opt.gen_vars


if flags.mom==2:
    #[top,bottom]
    axis_multiplier = {
        'gen_jet_ncharged':[1.6,0.5], 
        'gen_jet_charge':[1.4,0.8], 
        'gen_jet_ptD':[1.4,0.8],
        'gen_jet_tau10':[1.55,0.6], 
        'gen_jet_tau15':[1.55,0.6],
        'gen_jet_tau20':[1.55,0.6],


        'jet_ncharged':[1.35,0.6], 
        'jet_charge':[1.35,0.8], 
        'jet_ptD':[1.3,0.95],
        'jet_tau10':[1.4,0.6], 
        'jet_tau15':[1.4,0.6],
        'jet_tau20':[1.3,0.55],
    }

elif flags.mom==1:
    axis_multiplier = {
        'gen_jet_ncharged':[1.4,0.65], 
        'gen_jet_charge':[1.7,0.05], 
        'gen_jet_ptD':[1.15,0.97],
        'gen_jet_tau10':[0.6,1.5], 
        'gen_jet_tau15':[0.55,1.5],
        'gen_jet_tau20':[0.55,1.5],

        'jet_ncharged':[1.3,0.7], 
        'jet_charge':[1.7,0.7], 
        'jet_ptD':[1.25,0.97],
        'jet_tau10':[0.65,1.4], 
        'jet_tau15':[0.6,1.4],
        'jet_tau20':[0.6,1.4],
    }

else:
    axis_multiplier = {
        'gen_jet_ncharged':[2.0,0.4], 
        'gen_jet_charge':[1.3,0.8], 
        'gen_jet_ptD':[1.2,0.9],
        'gen_jet_tau10':[1.4,0.8], 
        'gen_jet_tau15':[1.4,0.8],
        'gen_jet_tau20':[1.4,0.8],
    }

    
def weighted_moments(values, weights,mom=1):
    if mom == 'var':
        return np.sqrt(np.average(values**2, weights=weights) - np.average(values, weights=weights)**2)
    else:
        return  np.average(values**mom, weights=weights)


def LoadData(q2_int):
    mc_info = {}
    weights_data = {}
    sys_variations = {}
    mc_info['data'] = MCInfo('data',flags.N,flags.data_folder,config,q2_int,is_reco=True)  
    #Loading weights from training
    for mc_name in mc_names:
        # print("{}.h5".format(mc_name))    
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
    yScalarFormatter = opt.ScalarFormatterClass(useMathText=True)
    yScalarFormatter.set_powerlimits((1000,0))
    ax0.yaxis.set_major_formatter(yScalarFormatter)
    
    #ax0.tick_params(axis='x',labelsize=0)
    # ax1 = plt.subplot(gs[1],sharex=ax0)
    ax0.set_xscale('log')
    if flags.mom==2:
        opt.FormatFig(xlabel = r'$Q^2$ [GeV$^2$]', ylabel = r'<(%s)$^2$>'%gen_var_names[var],ax0=ax0,ypos=0.95)
    elif flags.mom==1:
        opt.FormatFig(xlabel = r'$Q^2$ [GeV$^2$]', ylabel = r'<%s>'%gen_var_names[var],ax0=ax0,ypos=0.95)
    else:
        opt.FormatFig(xlabel = r'$Q^2$ [GeV$^2$]', ylabel = r'Std(%s)'%gen_var_names[var],ax0=ax0,ypos=0.95)
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
            data_mom = np.average(data_var**flags.mom)
        else:
            data_var = mc_info[data_name].LoadVar(var)
            data_mom = weighted_moments(data_var,weights=(weight_data*mc_info[data_name].nominal_wgts),mom=flags.mom)
            
        data_pred.append(data_mom)
        #######################################
        # Processing systematic uncertainties #
        #######################################
        if flags.sys == True:
            
            if not flags.plot_reco:
                #Stat uncertainty
                straps = []
                for strap in range(len(sys_variations['stat'])):
                    sys_pred = weighted_moments(data_var,weights=sys_variations['stat'][strap]*mc_info[data_name].nominal_wgts,mom=flags.mom)
                    straps.append(sys_pred)
                stat_unc.append(np.std(straps,axis=0)/np.mean(straps,axis=0))
            else:
                stat_unc.append(np.sqrt(data_var.shape[0])/data_var.shape[0])
                
            total_sys= stat_unc[-1]**2 
            for unc in opt.sys_sources:
                if unc == 'stat': continue
                if unc == 'QED': continue
                if( unc == 'closure' or unc == 'model') and flags.plot_reco:continue
                elif unc == 'closure':
                    sys_pred = weighted_moments(data_var,weights=(sys_variations[unc]*mc_info[data_name].nominal_wgts)) #should look like djangoh
                    mc_var = mc_info[mc_ref].LoadVar(var)
                    mc_mom = weighted_moments(mc_var,weights=mc_info[mc_ref].nominal_wgts)
                    ref_pred = mc_mom
                    
                elif unc == 'model':
                    #Model uncertainty: difference between unfolded values
                    data_sys = mc_info[mc_ref].LoadVar(var)
                    sys_pred = weighted_moments(data_sys,weights=(sys_variations[unc]*mc_info[mc_ref].nominal_wgts),mom=flags.mom)
                    ref_pred = data_mom
                else:
                    data_sys = mc_info[unc].LoadVar(var)
                    if flags.plot_reco:
                        sys_pred = weighted_moments(data_sys,weights=mc_info[unc].nominal_wgts,mom=flags.mom)
                    else:
                        sys_pred = weighted_moments(data_sys,weights=(sys_variations[unc]*mc_info[unc].nominal_wgts),mom=flags.mom)
                    ref_pred = data_mom
                    
                ratio_sys = np.divide(sys_pred-ref_pred,ref_pred)


                
                #if np.abs(ratio_sys)>np.abs(stat_unc[-1]):
                total_sys+= np.abs(ratio_sys**2 - stat_unc[-1]**2)
                # else:
                #     total_sys+= ratio_sys**2
                    
                print(unc,ratio_sys,stat_unc[-1])
            print(np.sqrt(total_sys))
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
            mom = weighted_moments(mc_var,mc_info[mc_name].nominal_wgts,mom=flags.mom)
            mc_pred[mc_name].append(mom)

    if not flags.plot_reco:
        for mc_name in add_predictions:
            if flags.mom==2:
                mc_pred[mc_name]= add_predictions[mc_name][var]['mom2']
            elif flags.mom==1:
                mc_pred[mc_name]= add_predictions[mc_name][var]['mom1']
            else:
                mc_pred[mc_name]= np.sqrt(np.array(add_predictions[mc_name][var]['mom2'])-np.array(add_predictions[mc_name][var]['mom1'])**2)


    #Lets plot the results
    data_pred = np.array(data_pred)
    #ax0.set_ylim(top=2.5*max(data_pred),bottom = 0.5*min(data_pred))
    ax0.set_xlim(left=95,right=5.5e3)
    
    if max(data_pred) <0:
        ax0.set_ylim(top=axis_multiplier[var][0]*min(data_pred),bottom = axis_multiplier[var][1]*max(data_pred))
    else:
        ax0.set_ylim(top=axis_multiplier[var][0]*max(data_pred),bottom = axis_multiplier[var][1]*min(data_pred))

    plt.text(text_xpos, text_ypos,'$Q^{2}>$150 GeV$^{2}$ \n $0.2<y<0.7$ \n $p_\mathrm{T}^\mathrm{jet}>10$ GeV  \n $k_\mathrm{T}, R=1.0$',
             horizontalalignment='center',
             verticalalignment='center',
             transform = ax0.transAxes, fontsize=18)
    #print("all uncs",np.sqrt(sys_unc))
    ax0.errorbar(xaxis,data_pred,yerr = np.abs(data_pred)*np.sqrt(sys_unc),fmt='o',ms=12,color='k',label='Data')
    for ibin in range(len(xaxis)):
        xup = binning[ibin+1]
        xlow = binning[ibin]
        ax0.hlines(y=data_pred[ibin], xmin=xlow, xmax=xup, colors='black')
        # ax0.fill_between(np.array([xlow,xup]),data_pred[ibin]+np.sqrt(sys_unc[ibin]),data_pred[ibin]-np.sqrt(sys_unc[ibin]), alpha=0.3,color='k')

    
    for mc_name in mc_pred:
        if 'nominal' in mc_name:
            mc = mc_name.split("_")[0]
        else:
            mc = mc_name
        displacement = opt.xaxis_disp[mc]
        xpos = xaxis+np.abs(np.diff(binning)/2.)*displacement
        ax0.plot(xpos,mc_pred[mc_name],color=opt.colors[mc],marker=opt.markers[mc],ms=10,lw=0,markerfacecolor='none',markeredgewidth=3,label=opt.name_translate[mc])
        
    ax0.legend(loc='upper left',fontsize=12,ncol=2)    


    base_folder = "plots_mom{}".format(flags.mom)
    if flags.plot_reco:
        base_folder += "_reco"
        
    plot_folder = os.path.join('..',"{}_{}".format(base_folder,data_name))    
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
    moment = "mom{}".format(flags.mom)
    fig.savefig(os.path.join(plot_folder,"{}_{}_{}.{}".format(var,flags.niter,moment,flags.img_fmt)))
