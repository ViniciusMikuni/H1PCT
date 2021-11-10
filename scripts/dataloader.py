import os
import uproot3
import pandas as pd
import numpy as np
import awkward

def get_Dataframe(path, name='Data', tag=None, pct = False,verbose=False):
    Files = os.listdir(path) 
    #print (Files)
    df = None
    for i, f in enumerate(Files):
        if name not in f: continue
        if 'Eplus0607_1' not in f:continue
        #if 'Eplus0607_144' not in f:continue
        if not f.endswith(".root"): continue
        
        filename = os.path.join(path,f)
        if not(tag is None) and (tag not in f): continue
        if (verbose):
            print ('filename is' , filename)
        
        temp_file = uproot3.open(filename)        
        hasTree = False 

        if (verbose):
            print (temp_file.keys()) 
                    
        if(len(temp_file.keys())<1):
            if (verbose):
                print('could not find %s, skipping'%name)
            continue
        
        if( not(name in str(temp_file.keys()[0]))):
            if (verbose):
                print('could not find %s, skipping'%name)
            continue

        try:
            temp_tree = temp_file[name+'/minitree']
        except:
            print("ERROR:TTree not found in file {}".format(filename))
            continue

        
        temp_df = None
        branches = [branch.decode('UTF-8') for branch in temp_tree.keys() if 'part' not in str(branch)]
        if verbose:print(branches)



        try:
            temp_df   =  temp_tree.pandas.df(branches, entrystop=3e7,flatten=True)
            if pct:            
                part_branches = [branch.decode('UTF-8') for branch in temp_tree.keys() if 'jet_part' in str(branch)]

                parts = temp_tree.arrays(part_branches)
                parts = SplitJets(parts) #One entry per jet   
                for ik, key in enumerate(parts.keys()):
                    temp_df[key] = parts[key].pad(20, clip=True).fillna(0).regular().tolist()
                
            df = pd.concat([df,temp_df])
        except ValueError:
            if (verbose):
                print ('oops, there is a problem in flattening the TTree ')
            
    if (verbose):
        print('####################################################################')
        if( not(df is None)):
            print('Dataframe has a total of ', df.shape[0], ' entries')
        else:
            print ('Dataframe has no entry, it is None')
        print('####################################################################')

    return df

def applyCut(inputDataframe, cut, text=None,verbose=False):
    dataframe = inputDataframe
    nbeforecut = dataframe.shape[0]
    cutDataframe = dataframe.query(cut)
    if text and verbose:
        print (text, cutDataframe.shape[0], ' fraction kept: %2.1f'%(100.0*float(cutDataframe.shape[0])/nbeforecut))
    return cutDataframe

def applyCutsJets(df,isMC=False,pct = False,verbose=False):
    temp = df

    temp['pass_reco'] = np.where(temp['jet_pt']>0, 1, 0)
    if (isMC):
        temp['pass_truth'] = np.where(temp['genjet_pt']*temp['gen_Q2']>0, 1, 0)
        temp['pass_fiducial'] = np.where(temp['pass_truth']*(temp['gen_Q2'] > 150)*
                                         (temp['gen_y']>0.2)*(temp['gen_y']<0.7)*
                                         (temp['genjet_pt']>10)*
                                         (temp['genjet_eta']<2.5)*
                                         (temp['genjet_eta']>-1.), 1, 0)
        
    #temp = applyCut(temp, 'abs(vertex_z)<25 and vertex_z!=0','abs(vertex_z)<25 and and vertex_z!=0')
    #temp = applyCut(temp, 'tau1b>0 and tau1b<1', '0<tau1b<1')
    temp.eval('jet_px = jet_pt*cos(jet_phi)', inplace=True)
    temp.eval('jet_py = jet_pt*sin(jet_phi)', inplace=True)
    temp.eval('jet_pz = jet_pt*sinh(jet_eta)', inplace=True)

    temp.eval('jet_qt = sqrt( (jet_px + e_px)**2 + (jet_py + e_py)**2) ', inplace=True)
    temp.eval('jet_qtnorm = jet_qt/sqrt(Q2)', inplace=True)
    temp.eval('e_pt = sqrt(e_px*e_px + e_py*e_py)',inplace=True)
    temp.eval('e_phi = arctan(e_py/e_px)', inplace=True)
    temp.eval('jet_phi = arctan(jet_py/jet_px)',inplace=True)
    
    temp.eval('qt_px = jet_px + e_px', inplace=True)
    temp.eval('qt_py = jet_py + e_py', inplace=True)
    temp.eval('qt_phi = arctan(qt_py/qt_px)',inplace=True)
    temp.eval('qt_dot_ept = (qt_px*e_px + qt_py*e_py)/(jet_qt*e_pt)', inplace=True)
    temp.eval('qt_dphi = arccos(qt_dot_ept)', inplace=True)
    temp.eval('qt_cos2phi = cos(2*qt_dphi)', inplace=True)

    temp.eval('jet_dphi = abs(e_phi-jet_phi)',inplace=True)
    temp.eval('logQ2= log(Q2)/2.3025850', inplace=True)
    temp.eval('Q = sqrt(Q2)', inplace=True)
    temp.eval('pthoverpte = pth/e_pt', inplace=True)
    temp = applyCut(temp, 'pass_reco==0 | ptmiss < 10', 'ptmiss<10',verbose)

    temp = applyCut(temp, 'pass_reco==0 | 0.08 < y < 0.7', '0.08 < y < 0.7',verbose)
    temp = applyCut(temp, 'pass_reco==0 | Q2>150', 'Q2>150',verbose)
   # temp = applyCut(temp, 'pass_reco==0 | Q2<10000', 'Q2<10000')
    temp = applyCut(temp, 'pass_reco==0 | Empz<65', 'Empz<65',verbose)
    temp = applyCut(temp, 'pass_reco==0 | Empz>45', 'Empz>45',verbose)
    temp = applyCut(temp, 'pass_reco==0 | jet_pt>5.0', 'jet pT > 5 GeV',verbose)
    temp = applyCut(temp, 'pass_reco==0 | jet_pt<150.0', 'jet pT < 150 GeV',verbose)

    temp = applyCut(temp, 'pass_reco==0 | jet_eta>-1.5', 'jet eta > -1.5',verbose)
    temp = applyCut(temp, 'pass_reco==0 | jet_eta<2.75', 'jet eta < 2.75',verbose)

    if(isMC):
        temp = applyCut(temp,'pass_truth>0',' pass_truth>0',verbose)

        temp.eval('gen_logQ2= log(gen_Q2)/2.3025850', inplace=True)   
        temp.eval('gen_Q    = sqrt(gen_Q2)', inplace=True)
        temp.eval('gene_pt = sqrt(gene_px*gene_px + gene_py*gene_py)',inplace=True)
        temp.eval('genjet_px = genjet_pt*cos(genjet_phi)', inplace=True)
        temp.eval('genjet_py = genjet_pt*sin(genjet_phi)', inplace=True)
        temp.eval('genjet_pz = genjet_pt*sinh(genjet_eta)', inplace=True)

        
        temp.eval('genjet_qt = sqrt( (genjet_px + gene_px)**2 + (genjet_py + gene_py)**2) ', inplace=True)
        temp.eval('genjet_qtnorm = genjet_qt/sqrt(gen_Q2)', inplace=True)
        temp.eval('gene_phi = arctan(gene_py/gene_px)', inplace=True)
        temp.eval('genjet_phi = arctan(genjet_py/genjet_px)',inplace=True)
        temp.eval('genjet_dphi = abs(gene_phi-genjet_phi)',inplace=True)
        
        temp.eval('genqt_px = genjet_px + gene_px', inplace=True)
        temp.eval('genqt_py = genjet_py + gene_py', inplace=True)
        temp.eval('genqt_phi = arctan(genqt_py/genqt_px)',inplace=True)
        temp.eval('genqt_dot_ept = (genqt_px*gene_px + genqt_py*gene_py)/(genjet_qt*gene_pt)', inplace=True)
        temp.eval('genqt_dphi = arccos(genqt_dot_ept)', inplace=True)
        temp.eval('genqt_cos2phi = cos(2*genqt_dphi)', inplace=True)

    #    temp.eval('genjet_qtnormept= genjet_qt/e_pt', inplace=True)
    #    temp.eval('genjet_qtnormjetpt= genjet_qt/genjet_pt', inplace=True)

    #Save only the features we need.
    feature_list = [
        'e_px','e_py','e_pz',
        'jet_pt','jet_phi','jet_eta','Q2',
        'jet_ncharged', 'jet_charge', 'jet_ptD','jet_tau10', 'jet_tau15', 'jet_tau20',
        'wgt','pass_reco']
    if (isMC):
        feature_list += [
            'gene_px','gene_py','gene_pz',
            'genjet_pt','genjet_phi','genjet_eta',
            'gen_jet_ncharged', 'gen_jet_charge', 'gen_jet_ptD',
            'gen_jet_tau10', 'gen_jet_tau15', 'gen_jet_tau20',
            'gen_Q2',
            'pass_truth', 'pass_fiducial']
        if pct:
            feature_list += ['gen_jet_part_pt','gen_jet_part_eta','gen_jet_part_phi','gen_jet_part_charge']
    if pct:
        feature_list += ['jet_part_pt','jet_part_eta','jet_part_phi','jet_part_charge']

    return temp[feature_list]


def SplitJets(data_dict):
    new_dict = {}
    idx = data_dict[b'jet_part_idx']
    print(idx)
    gen_idx = data_dict[b'gen_jet_part_idx']
    nparts = []
    nparts_gen = []
    for ie,entry in enumerate(idx):
        ninvalid = np.sum(entry<0)
        entry[entry<0]=range(100,100+ninvalid)
        
        for unique in np.unique(entry):
            nparts.append(np.sum(unique==entry))
        for unique in np.unique(gen_idx[ie]):
            nparts_gen.append(np.sum(unique==gen_idx[ie]))
                
            
    for key in data_dict:
        if 'gen' in key.decode('UTF-8'):
            new_dict[key.decode('UTF-8')] = awkward.JaggedArray.fromcounts(nparts_gen,data_dict[key].flatten())
        else:
            new_dict[key.decode('UTF-8')] = awkward.JaggedArray.fromcounts(nparts,data_dict[key].flatten())
    return new_dict
