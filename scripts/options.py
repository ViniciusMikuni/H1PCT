import numpy as np

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
    'MLP':'blueviolet',
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
    'MLP':'o',
}

# dedicated_binning = {
#     'gen_jet_ncharged':np.linspace(0,15,8), 
#     'gen_jet_charge':np.linspace(-1,1,7),
#     'genjet_pt': np.logspace(np.log10(10),np.log10(100),7),
#     'genjet_eta':np.linspace(-1,2.5,6),
#     'genjet_phi':np.linspace(-3.14,3.14,8),

#     'gen_jet_tau10':np.linspace(-3,-1,7),
#     'gen_jet_tau15':np.linspace(-3,-1,7),
#     'gen_jet_tau20':np.linspace(-4,-1,7),


#     'gen_jet_ptD':np.linspace(0,1,6),


#     'gene_px':np.logspace(np.log10(1),np.log10(100),7),
#     'gene_py':np.logspace(np.log10(1),np.log10(100),7),
#     'gene_pz':np.logspace(np.log10(1),np.log10(100),7),

#     # 'gen_jet_tau10':np.linspace(0,1,7),
#     # 'gen_jet_tau15':np.linspace(0,1,7),
#     # 'gen_jet_tau20':np.linspace(0,1,7),


# }


dedicated_binning = {
    'gen_jet_ncharged':np.linspace(1,15,15),
    'gen_jet_charge':np.linspace(-1,1,30),
    'genjet_pt': np.logspace(np.log10(10),np.log10(100),30),
    'gen_Q2': np.logspace(np.log10(150),np.log10(1000),30),
    'genjet_eta':np.linspace(-1,2.5,6),
    'genjet_phi':np.linspace(-3.14,3.14,8),

    'gen_jet_tau10':np.linspace(-3,-0.1,30),
    'gen_jet_tau15':np.linspace(-4,-0.5,30),
    'gen_jet_tau20':np.linspace(-5,-0.5,30),


    'gen_jet_ptD':np.linspace(0.2,1,30),


    'gene_px':np.logspace(np.log10(1),np.log10(100),7),
    'gene_py':np.logspace(np.log10(1),np.log10(100),7),
    'gene_pz':np.logspace(np.log10(1),np.log10(100),7),
}



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
