import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

colors = {
    'LO':'b', 
    'NLO':'g',
    'NNLO':'r', 
    'TMD': '#9467bd',
    'Pythia':'blueviolet',
    'Djangoh':'#8c564b',
    'Rapgap':'darkorange',
    'Sherpa':'crimson',
    'Cascade':'b',
    
    'PCT': 'g',    
    'standard':'blueviolet',
    'hybrid':'red',
}



styles = {
    'LO':'-',
    'NLO':'dotted',
    'NNLO':'-',
    'TMD': '-',
    'Pythia':'-',
    'Djangoh':'dotted',
    'Rapgap':'-',
    'Sherpa':'-',
    'Cascade':'-',    
}

markers = {
    'Djangoh':'D',
    'Rapgap':'X',
    
    'PCT':'P',
    'standard':'o',
    'hybrid':'x',
}

dedicated_binning = {
    'gen_jet_ncharged':np.linspace(1,11,10), 
    'gen_jet_charge':np.linspace(-0.9,0.9,10),
    'genjet_pt': np.logspace(np.log10(10),np.log10(100),7),
    'genjet_eta':np.linspace(-1,2.5,6),
    'genjet_phi':np.linspace(-3.14,3.14,8),
    'gen_jet_tau10':np.linspace(-2.2,-1,10),
    'gen_jet_tau15':np.linspace(-3,-1,10),
    'gen_jet_tau20':np.linspace(-3.5,-1,10),

    'gen_jet_ptD':np.linspace(0.2,0.7,10),
    'gen_Q2':np.logspace(np.log10(200),np.log10(1000), 5),
}

sys_sources = {
    'sys_0':'#66c2a5',
    'sys_1':'#fc8d62',
    'sys_5':'#a6d854',
    'sys_7':'#ffd92f',
    'sys_11':'#8da0cb',
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
    'closure': 'Non-closure',
    'stat':'Stat.',
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

def SetGrid(ratio=True):
    fig = plt.figure(figsize=(9, 9))
    if ratio:
        gs = gridspec.GridSpec(2, 1, height_ratios=[3,1]) 
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
        

    xposition = 0.9
    yposition=1.03
    text = 'H1'
    WriteText(xposition,yposition,text,ax0)


def WriteText(xpos,ypos,text,ax0):

    plt.text(xpos, ypos,text,
             horizontalalignment='center',
             verticalalignment='center',
             transform = ax0.transAxes, fontsize=25, fontweight='bold')


    
