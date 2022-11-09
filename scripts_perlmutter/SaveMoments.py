import numpy as np
import json, yaml
import h5py as h5
import numpy as np
import uproot3 as uproot
import sys,os
sys.path.append('../')
import shared.options as opt


def SaveJson(save_file,data):
    with open(save_file,'w') as f:            
        json.dump(data, f)

def weighted_moments(values, weights):
    average = np.average(values, weights=weights)
    moment2 = np.average(values**2, weights=weights)
    return (average, moment2)


file_dict={
    'Pythia_Vincia': 'pythia83_vincia_10M_118_tree.root',
    'Pythia_Dire': 'pythia83_dire_1M_tree.root',
    'Pythia':'pythia83_default_10M_118_tree.root',
}

base_path = '/global/cfs/cdirs/m3929/H1/'
var_list = opt.gen_vars
q2_bin = opt.dedicated_binning['gen_Q2']
moments = {}
save_file='jet_substructure_moments.json'


for mc in file_dict:
    moments[mc] = {}
    file_path=os.path.join(base_path,file_dict[mc])
    tree= uproot.open(file_path)['Tree']
    q2 = tree['Q2'].array()
    wgt = tree['weight'].array()
    for var in var_list:
        moments[mc][var]={'mom1':(len(q2_bin)-1)*[0],'mom2':(len(q2_bin)-1)*[0]}
        var_array = tree[var].array()
        for q2_int in range(len(q2_bin)-1):
            mask = (q2>q2_bin[q2_int]) & (q2<q2_bin[q2_int+1])
            mean,mom2 = weighted_moments(var_array[mask],wgt[mask])
            moments[mc][var]['mom1'][q2_int]=mean
            moments[mc][var]['mom2'][q2_int]=mom2

#Load moments from root files that were already preprocessed
base_path = '/global/homes/v/vmikuni/H1/H1PCT/rivet'
rivet_predictions = {
    # 'Sherpa3NLO': 'Sherpa3NLO.root',
    # 'Sherpa2LOLund': 'Sherpa2LOLund.root',
    # 'Sherpa2LOCluster': 'Sherpa2LOCluster.root',
    'Sherpa2LOCluster': 'Analysis_DIS_JetSub.root',
    'Herwig':'JetSubsNominal10-23-22.root',
    'Herwig_Merging':'JetSubsMerging10-23-22.root',
    'Herwig_Matchbox':'JetSubsMatchbox10-23-22.root',
}
for mc in rivet_predictions:
    moments[mc] = {}
    file_path=os.path.join(base_path,rivet_predictions[mc])
    hists= uproot.open(file_path)
    for var in var_list:
        #print(hists[var+'_mom1'].values)
        moments[mc][var]={
            'mom1':list(hists[var+'_mom1'].values),
            'mom2':list(hists[var+'_mom2'].values)}
            
SaveJson(save_file,moments)

