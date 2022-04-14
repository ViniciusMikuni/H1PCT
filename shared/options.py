import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import uproot3 as uproot
    
colors = {
    'LO':'b', 
    'NLO':'g',
    'NNLO':'r', 
    'Pythia_Vincia': '#9467bd',
    'Pythia_Dire': 'indigo',
    'Pythia':'blueviolet',
    'Djangoh':'#8c564b',
    'Rapgap':'darkorange',
    'Herwig':'crimson',
    'Herwig_Matchbox':'crimson',
    'Cascade':'b',
    
    'PCT': 'g',    
    'standard':'blueviolet',
    'hybrid':'red',
}


markers = {
    'Djangoh':'P',
    'Rapgap':'X',

    'Pythia': '^',
    'Pythia_Vincia': '<',
    'Pythia_Dire': '>',
    'Herwig':'D',
    'Herwig_Matchbox':'d',
    'PCT':'P',
    'standard':'o',
    'hybrid':'x',
}

#Shift in x-axis for visualization
xaxis_disp = {
    'Pythia': 0.0,
    'Pythia_Vincia': 0.3,
    'Pythia_Dire': 0.6,
    'Herwig':-0.3,
    'Herwig_Matchbox':-0.6,
    'Djangoh':-0.3,
    'Rapgap':0.3
}
    

dedicated_binning = {
    'gen_jet_ncharged':np.linspace(1,15-1e-8,15), 
    'gen_jet_charge':np.linspace(-0.8,0.8,10),
    'genjet_pt': np.logspace(np.log10(10),np.log10(100),7),
    'genjet_eta':np.linspace(-1,2.5,6),
    'genjet_phi':np.linspace(-3.14,3.14,8),
    
    'gen_jet_tau10':np.linspace(-2.2,-1,8),
    'gen_jet_tau15':np.linspace(-3,-1.2,8),
    'gen_jet_tau20':np.linspace(-3.5,-1.5,8),
    'gen_jet_ptD':np.linspace(0.3,0.7,10),
    'gen_Q2':np.logspace(np.log10(150),np.log10(5000), 5),

    'jet_ncharged':np.linspace(1,15-1e-8,15), 
    'jet_charge':np.linspace(-0.8,0.8,10),    
    'jet_tau10':np.linspace(-2.2,-1,8),
    'jet_tau15':np.linspace(-3,-1.2,8),
    'jet_tau20':np.linspace(-3.5,-1.5,8),
    'jet_ptD':np.linspace(0.3,0.7,10),
    'Q2':np.logspace(np.log10(150),np.log10(5000), 5),
}

fixed_yaxis = {
    'gen_jet_ncharged':0.45, 
    'gen_jet_charge':4.5,
    'gen_jet_tau10':3,
    'gen_jet_tau15':2,
    'gen_jet_tau20':2,
    'gen_jet_ptD':12,

    }

sys_sources = {
    'sys_0':'#66c2a5',
    'sys_1':'#fc8d62',
    'sys_5':'#a6d854',
    'sys_7':'#ffd92f',
    'sys_11':'#8da0cb',
    'QED': '#8c564b',
    'model':'#e78ac3',
    'closure': '#e5c494',
    'stat':'#808000'

}

sys_translate = {
    'sys_0':"HFS scale (in jet)",
    'sys_1':"HFS scale (remainder)",
    'sys_5':"HFS $\phi$ angle",
    'sys_7':"Lepton energy scale",
    'sys_11':"Lepton $\phi$ angle",
    'model': 'Model',
    'QED':'QED correction',
    'closure': 'Non-closure',
    'stat':'Stat.',
}

name_translate = {
    'Herwig': "Herwig",
    'Herwig_Matchbox': "Herwig + Matchbox",
    'Pythia': 'Pythia',
    'Pythia_Vincia':'Pythia + Vincia',
    'Pythia_Dire':'Pythia + Dire',

}

reco_vars = {
    'jet_ncharged':r'Charged hadron multiplicity $(\tilde{\lambda}_0^0)$', 
    'jet_charge':r'Jet Charge $(\tilde{\lambda}_0^1)$', 
    'jet_ptD':r'$p_\mathrm{T}\mathrm{D}$ $(\sqrt{\lambda_0^2})$',
    'jet_tau10':r'$\mathrm{log}(\lambda_1^1)$', 
    'jet_tau15':r'$\mathrm{log}(\lambda_{1.5}^1)$',
    'jet_tau20':r'$\mathrm{log}(\lambda_2^1)$',
}

gen_vars = {
    'gen_jet_ncharged':r'Charged hadron multiplicity $(\tilde{\lambda}_0^0)$', 
    'gen_jet_charge':r'Jet Charge $(\tilde{\lambda}_0^1)$', 
    'gen_jet_ptD':r'$p_\mathrm{T}\mathrm{D}$ $(\sqrt{\lambda_0^2})$',
    'gen_jet_tau10':r'$\mathrm{log}(\lambda_1^1)$', 
    'gen_jet_tau15':r'$\mathrm{log}(\lambda_{1.5}^1)$',
    'gen_jet_tau20':r'$\mathrm{log}(\lambda_2^1)$',
}

# dedicated_binning = {
#     'gen_jet_ncharged':np.linspace(1,15,14),
#     'gen_jet_charge':np.linspace(-1,1,30),
#     'genjet_pt': np.logspace(np.log10(10),np.log10(100),30),
#     'jet_pt': np.logspace(np.log10(10),np.log10(100),30),
#     'genjet_eta':np.linspace(-1,2.5,6),
#     'genjet_phi':np.linspace(-3.14,3.14,8),

#     'gen_jet_tau10':np.linspace(-3,-0.3,30),
#     'gen_jet_tau15':np.linspace(-4,-0.5,30),
#     'gen_jet_tau20':np.linspace(-5,-0.5,30),
    
    
#     'gen_jet_ptD':np.linspace(0.2,0.9,30),
#     'gen_Q2':np.logspace(np.log10(200),np.log10(1000),5),
    
    
#     'gene_px':np.logspace(np.log10(1),np.log10(100),7),
#     'gene_py':np.logspace(np.log10(1),np.log10(100),7),
#     'gene_pz':np.logspace(np.log10(1),np.log10(100),7),
# }


def LoadFromROOT(file_name,var_name,q2_bin=0):
    with uproot.open(file_name) as f:
        if b'DIS_JetSubs;1' in f.keys():
            #Predictions from rivet
            hist = f[b'DIS_JetSubs;1']            
        else:
            hist = f
        if q2_bin ==0:
            var, bins =  hist[var_name].numpy()            
        else: #2D slice of histogram
            var =  hist[var_name+"2D"].numpy()[0][:,q2_bin-1]
            bins = hist[var_name+"2D"].numpy()[1][0][0]
            
        norm = 0
        for iv, val in enumerate(var):
            norm += val*abs(bins[iv+1]-bins[iv])
        return var
        
def SetStyle():
    from matplotlib import rc
    rc('text', usetex=True)

    import matplotlib as mpl
    rc('font', family='serif')
    rc('font', size=22)
    rc('xtick', labelsize=15)
    rc('ytick', labelsize=15)
    rc('legend', fontsize=15)

    # #
    mpl.rcParams.update({'font.size': 19})
    #mpl.rcParams.update({'legend.fontsize': 18})
    mpl.rcParams['text.usetex'] = False
    mpl.rcParams.update({'xtick.labelsize': 18}) 
    mpl.rcParams.update({'ytick.labelsize': 18}) 
    mpl.rcParams.update({'axes.labelsize': 18}) 
    mpl.rcParams.update({'legend.frameon': False}) 
    
    import matplotlib.pyplot as plt
    import mplhep as hep
    hep.set_style(hep.style.CMS)
    hep.style.use("CMS") 

def SetGrid(npanels=2):
    fig = plt.figure(figsize=(9, 9))
    if npanels ==2:
        gs = gridspec.GridSpec(2, 1, height_ratios=[3,1]) 
        gs.update(wspace=0.025, hspace=0.1)
    elif npanels ==3:
        gs = gridspec.GridSpec(3, 1, height_ratios=[3,1,1]) 
        gs.update(wspace=0.025, hspace=0.1)
    else:
        gs = gridspec.GridSpec(1, 1)
    return fig,gs

def FormatFig(xlabel,ylabel,ax0):
    #Limit number of digits in ticks
    # y_loc, _ = plt.yticks()
    # y_update = ['%.1f' % y for y in y_loc]
    # plt.yticks(y_loc, y_update) 
    ax0.set_xlabel(xlabel,fontsize=20)
    ax0.set_ylabel(ylabel)
        

    xposition = 0.8
    yposition=0.9
    # xposition = 0.83
    # yposition=1.03
    text = r'$\bf{H1}$ Preliminary'
    WriteText(xposition,yposition,text,ax0)


def WriteText(xpos,ypos,text,ax0):

    plt.text(xpos, ypos,text,
             horizontalalignment='center',
             verticalalignment='center',
             #fontweight='bold',
             transform = ax0.transAxes, fontsize=25)


    

def make_error_boxes(ax, xdata, ydata, xerror, yerror, facecolor='r',
                     edgecolor='None', alpha=0.5):

    # Loop over data points; create box from errors at each point
    errorboxes = [Rectangle((x - xe[0], y - ye[0]), xe.sum(), ye.sum())
                  for x, y, xe, ye in zip(xdata, ydata, xerror.T, yerror.T)]

    # Create patch collection with specified colour/alpha
    pc = PatchCollection(errorboxes, facecolor=facecolor, alpha=alpha,
                         edgecolor=edgecolor)

    # Add collection to axes
    ax.add_collection(pc)

    # Plot errorbars
    artists = ax.errorbar(xdata, ydata, xerr=xerror, yerr=yerror,
                          fmt='None', ecolor='k')
