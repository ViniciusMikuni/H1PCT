from yoda.script_helpers import parse_x2y_args
import numpy as np
import json, yaml
import sys,os, yoda
import argparse
sys.path.append('../')


parser = argparse.ArgumentParser(usage=__doc__)
parser.add_argument("ARGS", nargs="+", help="infile [outfile]")

args = parser.parse_args()
in_out = parse_x2y_args(args.ARGS, ".yoda", ".root")
if not in_out:
    sys.stderr.write("You must specify the YODA and ROOT file names\n")
    sys.exit(1)

    
try:
    import ROOT
    ROOT.gROOT.SetBatch(True)
except ImportError:
    sys.stderr.write("Could not load ROOT Python module, exiting...\n")
    sys.exit(2)


def GetQ2Norm(hists,mom=1):
    for hist in hists:
        if 'Q2' in hist.name:
            if mom==1:
                values = [b.sumW for b in hist.bins]
            else:
                values = [b.sumW2 for b in hist.bins]
            return values

def GetCountNorm(hists,name,mom=1):
    for hist in hists:
        if name in hist.name:
            if mom==1:
                values = [b.numEntries for b in hist.bins]
            else:
                values = [b.numEntries for b in hist.bins]
            return values

root_hists = []
for i, o in in_out:
    aos = yoda.read(i)
    hs = [h for h in aos.values() if "DIS_JetSubs" in h.path]

    for h in hs:
        if 'RAW' in h.path:continue
        if 'tau' in h.name and 'log' not in h.name:continue
        print(h.name)
        values = [b.sumW for b in h.bins]
            
        if 'mom' in h.name:
            norm = GetQ2Norm(hs,mom=1)
            # norm1 = GetCountNorm(hs,h.name,mom=1)
            # norm2 = GetCountNorm(hs,h.name,mom=2) 
            values = np.divide(values,norm)
            #print(values)

        if '2D' in h.name:
            # print(h.name)
            values = np.array(values).reshape((len(h.xEdges())-1,len(h.yEdges())-1))
            root_hists.append(ROOT.TH2D(h.name.replace('log',''),h.title,len(h.xEdges())-1,h.xEdges(),len(h.yEdges())-1,h.yEdges())) #FIXME: Dumb way to separate log from linear
            for ibinx in range(len(h.xEdges())-1):
                for ibiny in range(len(h.yEdges())-1):
                    root_hists[-1].SetBinContent(ibinx+1,ibiny+1,values[ibinx,ibiny])

        else:
            # print(h.name)
            root_hists.append(ROOT.TH1D(h.name.replace('log',''),h.title,len(h.xEdges())-1,h.xEdges())) #FIXME: Dumb way to separate log from linear
            nbins= len(h.xEdges())-1

            for ibin in range(nbins):
                root_hists[-1].SetBinContent(ibin+1,values[ibin])


    # print(root_hists)
    of = ROOT.TFile(o, "recreate")
    for hist in root_hists:
        # print(hist.GetName())
        hist.Write()
