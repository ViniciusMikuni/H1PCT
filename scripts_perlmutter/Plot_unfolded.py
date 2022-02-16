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
parser.add_argument('--closure', action='store_true', default=False,help='Plot closure results')
parser.add_argument('--sys', action='store_true', default=False,help='Evaluate results with systematic uncertainties')
parser.add_argument('--dctr', action='store_true', default=False,help='Display DCTR correction for reco distributions')
parser.add_argument('--comp', action='store_true', default=False,help='Compare closure unc. from different methods')
parser.add_argument('-N',type=float,default=20e6, help='Number of events to evaluate')
parser.add_argument('--niter', type=int, default=5, help='Omnifold iteration to load')
parser.add_argument('--q2_int', type=int, default=0, help='Q2 interval to consider')

flags = parser.parse_args()
flags.N = int(flags.N)

config=LoadJson(flags.config)

mc_names = ['Rapgap_nominal','Djangoh_nominal']
    
data_idx = 1 #Sample that after weights represent data
data_name = mc_names[data_idx]

mc_ref = mc_names[data_idx-1]
print(mc_ref)
folder = 'results'

version = data_name
if flags.closure:
    version  +='_closure'
    

gen_var_names = {
    #'jet_pt': r"$p_\mathrm{T}^\mathrm{jet}$",
    'genjet_eta':r'$\eta^\mathrm{jet}$',
    'gen_Q2': r"$Q^2$",
    'genjet_pt': r"$p_\mathrm{T}^\mathrm{jet}$",
    # 'genjet_phi':r'$\phi^\mathrm{jet}$',
    'gen_jet_ncharged':r'$\mathrm{N_{c}}$', 
    'gen_jet_charge':r'$\mathrm{Q_j}$', 
    'gen_jet_ptD':r'$p_\mathrm{T}\mathrm{D}$',
    'gen_jet_tau10':r'$\mathrm{log}(\tau_{1})$', 
    'gen_jet_tau15':r'$\mathrm{log}(\tau_{0.5})$',
    'gen_jet_tau20':r'$\mathrm{log}(\tau_{0})$',

    # 'gene_px':'e px',
    # 'gene_py':'e px',
    # 'gene_pz':'e px',
}



def RatioLabel(ax1):
    ax1.set_ylabel('Rel. diff. [%]')
    ax1.set_xlabel(gen_var_names[var])    
    ax1.axhline(y=0.0, color='r', linestyle='-')
    # ax1.axhline(y=10, color='r', linestyle='--')
    # ax1.axhline(y=-10, color='r', linestyle='--')
    ax1.set_ylim([-30,30])
    

def PlotUnc(xaxis,values,xlabel='',add_text=''):
    fig = plt.figure(figsize=(9, 9))
    total_sys = np.zeros(len(xaxis))
    for sys in values:
        if 'stat' in sys:
            plt.plot(xaxis,np.abs(values[sys]),ls="--",lw=3,color=opt.sys_sources[sys],label=opt.sys_translate[sys])
            continue
        else:
            plt.plot(xaxis,np.abs(values[sys]),color=opt.sys_sources[sys],label=opt.sys_translate[sys])
        total_sys+=values[sys]**2
        
    plt.plot(xaxis,np.sqrt(total_sys),ls=":",color='black',label='Total Syst.',lw=3)    
    plt.xlabel(xlabel)
    plt.ylabel('Systematic uncertianty [%]')

    if len(add_text)>0:
        ax = plt.gca()
        opt.WriteText(0.3,1.03,add_text,ax)
    
    plt.legend(loc='best',fontsize=13,ncol=3)
    plt.ylim([0,20])
    return fig




mc_info = {}
weights_data = {}
sys_variations = {}


#Loading weights from training
for mc_name in mc_names:
    print("{}.h5".format(mc_name))    
    mc_info[mc_name] = MCInfo(mc_name,flags.N,flags.data_folder,config,flags.q2_int)        
    if mc_name == data_name:
        if flags.comp:
            for mode in config['COMPARISON']:
                print(mode)
                base_name = "Omnifold_{}".format(mode)
                model_name = '{}/{}_{}_iter{}_step2.h5'.format(flags.weights,base_name,version,flags.niter)
                weights_data[mode] = mc_info[mc_name].ReturnWeights(config['{}_NITER'.format(mode)],model_name=model_name,mode=mode)
        else:
            base_name = "Omnifold_{}".format(flags.mode)
            model_name = '{}/{}_{}_iter{}_step2.h5'.format(flags.weights,base_name,version,flags.niter)            
            weights_data[flags.mode] = mc_info[mc_name].ReturnWeights(flags.niter,model_name=model_name,mode=flags.mode)

            if flags.sys == True: #load systematic variations
                for sys in opt.sys_sources:
                    if sys in ['model','closure','stat']: continue
                    model_name = '{}/{}_{}_iter{}_step2.h5'.format(
                        flags.weights,base_name,version.replace("nominal",sys),flags.niter)
                    print(mc_name.replace("nominal",sys))
                    mc_info[sys] = MCInfo(mc_name.replace("nominal",sys),int(flags.N),flags.data_folder,config,flags.q2_int)
                    sys_variations[sys] = mc_info[sys].ReturnWeights(
                        flags.niter,model_name=model_name,mode=flags.mode)
                    
                #Load non-closure weights
                model_name = '{}/{}_{}_iter{}_step2.h5'.format(
                    flags.weights,base_name,version+'_closure',flags.niter)
                sys_variations['closure'] = mc_info[mc_name].ReturnWeights(
                    flags.niter,model_name=model_name,mode=flags.mode)
                
                sys_variations['stat'] = [
                    mc_info[mc_name].LoadTrainedWeights(
                        os.path.join(flags.data_folder,'weights','{}_{}.h5'.format(mc_name,nstrap))
                    ) for nstrap in range(1,config['NBOOTSTRAP']+1)]
                
    elif flags.sys:
        base_name = "Omnifold_{}".format(flags.mode)
        model_name = '{}/{}_{}_iter{}_step2.h5'.format(flags.weights,base_name,mc_name,flags.niter)
        sys_variations['model'] = mc_info[mc_name].ReturnWeights(
            flags.niter,model_name=model_name,mode=flags.mode)
            

weight_data = weights_data[flags.mode]


for var in gen_var_names:
    print(var)
    binning = opt.dedicated_binning[var]    
    data_var = mc_info[data_name].LoadVar(var)
    
    fig,gs = opt.SetGrid() 
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1],sharex=ax0)
    #xaxis = [(binning[i] + binning[i+1])/2.0 for i in range(len(binning)-1)]
    xaxis = 0.5*(binning[:-1] + binning[1:])
    xerr = 0.5*np.abs((binning[:-1] - binning[1:]))
    
    if 'genjet_pt' in var or 'Q2' in var:
        ax0.set_xscale('log')
        ax0.set_yscale('log')
        ax1.set_xscale('log')


        
    data_pred,_=np.histogram(data_var,weights=weight_data*mc_info[data_name].nominal_wgts,bins=binning,density=True)
        
    ratios = {}
    max_y = max(data_pred)
    ax0.set_ylim(top=1.5*max_y)
    
    if flags.sys == True:
        ratio_sys = {}
        total_sys = np.ones(len(binning)-1)

        #Stat uncertainty
        stat_unc = []
        for strap in range(len(sys_variations['stat'])):
            sys_pred,_ = np.histogram(data_var,weights=sys_variations[sys][strap]*mc_info[data_name].nominal_wgts,bins=binning,density=True)
            stat_unc.append(sys_pred)
        ratio_sys['stat'] = 100*np.std(stat_unc,axis=0)/np.mean(stat_unc,axis=0)
        total_sys+= ratio_sys['stat']**2
        #Plot data and stat error
        #print(xaxis,data_pred,np.abs(binning),data_pred*ratio_sys['stat']/100.)
        ax0.errorbar(xaxis,data_pred,yerr = data_pred*ratio_sys['stat']/100.,fmt='o',ms=12,color='k',label='Data')
        for ibin in range(len(xaxis)):
            xup = binning[ibin+1]
            xlow = binning[ibin]
            yup=data_pred[ibin]*(1 + ratio_sys['stat'][ibin]/100.)
            ylow=data_pred[ibin]*(1 - ratio_sys['stat'][ibin]/100.)
            ax0.fill_between(np.array([xlow,xup]),yup,ylow, alpha=0.3,color='k')
        
        
        for sys in opt.sys_sources:
            if sys == 'stat': continue
            if sys == 'model':
                #Model uncertainty: difference between unfolded values
                data_sys = mc_info[mc_ref].LoadVar(var)
                sys_pred,_ = np.histogram(data_sys,weights=sys_variations[sys]*mc_info[mc_ref].nominal_wgts,bins=binning,density=True)
                ratio_sys[sys] = 100*np.divide(sys_pred-data_pred,data_pred)
                ratio_sys[sys] = np.sqrt(np.abs(ratio_sys[sys]**2 - ratio_sys['stat']**2))
            elif sys == 'closure':
                sys_pred,_ = np.histogram(data_var,weights=sys_variations[sys]*mc_info[data_name].nominal_wgts,bins=binning,density=True)
                mc_var = mc_info[mc_ref].LoadVar(var)
                mc_pred,_ = np.histogram(mc_var,weights=mc_info[mc_ref].nominal_wgts,bins=binning,density=True)
                ratio_sys[sys] = 100*np.divide(sys_pred-mc_pred,mc_pred)
                ratio_sys[sys] = np.sqrt(np.abs(ratio_sys[sys]**2 - ratio_sys['stat']**2))
            else:
                data_sys = mc_info[sys].LoadVar(var)
                sys_pred,_ = np.histogram(data_sys,weights=sys_variations[sys]*mc_info[sys].nominal_wgts,bins=binning,density=True)            
                ratio_sys[sys] = 100*np.divide(sys_pred-data_pred,data_pred)
                ratio_sys[sys] = np.sqrt(np.abs(ratio_sys[sys]**2 - ratio_sys['stat']**2))
            total_sys+= ratio_sys[sys]**2
        
        for ibin in range(len(xaxis)):
            xup = binning[ibin+1]
            xlow = binning[ibin]
            ax1.fill_between(np.array([xlow,xup]),np.sqrt(total_sys[ibin]),-np.sqrt(total_sys[ibin]), alpha=0.3,color='k')
    else:
        data_pred,_,_=ax0.hist(data_var,weights=weight_data*mc_info[data_name].nominal_wgts,bins=binning,label="Data",density=True,color="black",histtype="step")
        
    opt.FormatFig(xlabel = "", ylabel = r'$1/\sigma$ $\mathrm{d}\sigma/\mathrm{d}$%s'%gen_var_names[var],ax0=ax0)
    if flags.q2_int>0:
        gen_q2 = opt.dedicated_binning['gen_Q2']        
        text_q2 = r"{} < Q$^2$ < {} GeV$^2$".format(int(np.round(gen_q2[flags.q2_int-1],1)),int(np.round(gen_q2[flags.q2_int],1)))
        opt.WriteText(xpos=0.25,ypos=1.03,text = text_q2,ax0=ax0)
        
        plt.text(0.15, 0.87,'$0.2<y<0.7$ \n $p_\mathrm{T}^\mathrm{jet}>10$ GeV  \n $k_\mathrm{T}, R=1.0$',
         horizontalalignment='center',
         verticalalignment='center',
         transform = ax0.transAxes, fontsize=20)
        
    else:
        plt.text(0.15, 0.77,'$Q^{2}>$150 GeV$^{2}$ \n $0.2<y<0.7$ \n $p_\mathrm{T}^\mathrm{jet}>10$ GeV  \n $k_\mathrm{T}, R=1.0$',
         horizontalalignment='center',
         verticalalignment='center',
         transform = ax0.transAxes, fontsize=20)
        
    ax0.tick_params(axis='x',labelsize=0)
    
    for mc_name in mc_names:
        #Upper canvas
        mc = mc_name.split("_")[0]
        mc_var = mc_info[mc_name].LoadVar(var)
        
        pred,_=np.histogram(mc_var,weights=mc_info[mc_name].nominal_wgts,bins=binning,density=True)
        ax0.plot(xaxis,pred,color=opt.colors[mc],marker=opt.markers[mc],ms=12,lw=0,markerfacecolor='none',markeredgewidth=3,label=mc)
        ratios[mc] = 100*np.divide(pred-data_pred,data_pred)        
        #Ratio plot    
        mc = mc_name.split("_")[0]
        ax1.plot(xaxis,ratios[mc],color=opt.colors[mc],marker=opt.markers[mc],ms=12,lw=0,markerfacecolor='none',markeredgewidth=3)


    ax0.legend(loc='best',fontsize=16,ncol=1)    
    RatioLabel(ax1)
    
    plot_folder = '../plots_'+data_name
    if flags.closure:
        plot_folder+='_closure'
    if flags.sys:
        plot_folder+='_sys'
    plot_folder+='_{}'.format(flags.mode)
    
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
    if flags.q2_int>0:
        fig.savefig(os.path.join(plot_folder,"{}_{}_{}.png".format(var,flags.niter,flags.q2_int)))
        if flags.sys:
            fig_sys = PlotUnc(xaxis,ratio_sys,gen_var_names[var],text_q2)
            fig_sys.savefig(os.path.join(plot_folder,"sys_{}_{}_{}.pdf".format(var,flags.niter,flags.q2_int)))
    else:
        fig.savefig(os.path.join(plot_folder,"{}_{}.pdf".format(var,flags.niter)))
        if flags.sys:
            fig_sys = PlotUnc(xaxis,ratio_sys,gen_var_names[var])
            fig_sys.savefig(os.path.join(plot_folder,"sys_{}_{}.pdf".format(var,flags.niter)))


    

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
        
        mc_var = mc_info[mc_ref].LoadVar(var)        
        mc_pred,_,_=ax0.hist(mc_var,weights=mc_info[mc_ref].nominal_wgts,bins=binning,label="Target Gen",density=True,color="black",histtype="step")

        opt.FormatFig(xlabel = "", ylabel = r'$1/\sigma$ $\mathrm{d}\sigma/\mathrm{d}$%s'%gen_var_names[var],ax0=ax0)
        if flags.q2_int>0:
            gen_q2 = opt.dedicated_binning['gen_Q2']      
            text = r"{} < Q$^2$ < {} GeV$^2$".format(int(np.round(gen_q2[flags.q2_int-1],1)),int(np.round(gen_q2[flags.q2_int],1)))
            opt.WriteText(xpos=0.1,ypos=1.03,text = text,ax0=ax0)

        ratios = {}
        
        data_var = mc_info[data_name].LoadVar(var)
        for train in weights_data:
            pred,_,_ = ax0.hist(data_var,weights=weights_data[train]*mc_info[data_name].nominal_wgts,bins=binning,
                                label=train,density=True,color=opt.colors[train],histtype="step")
            ratios[train] = 100*np.divide(mc_pred-pred,pred)
            
            ax1.plot(xaxis,ratios[train],color=opt.colors[train],marker=opt.markers[train],ms=12,lw=0,markerfacecolor='none',markeredgewidth=3)

            
        RatioLabel(ax1)
        ax0.legend(loc='lower right',fontsize=16,ncol=1)

        plot_folder = '../plots_comp_'+data_name
        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)


        if flags.q2_int>0:
            fig.savefig(os.path.join(plot_folder,"{}_{}_{}.pdf".format(var,flags.niter,flags.q2_int)))
        else:
            fig.savefig(os.path.join(plot_folder,"{}_{}.pdf".format(var,flags.niter)))








