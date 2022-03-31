// main36.cc is a part of the PYTHIA event generator.
// Copyright (C) 2017 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL version 2, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Basic setup for Deeply Inelastic Scattering at HERA.
#include <iostream>
#include <vector>
#include <sstream>
#include <string>

#include "TParticlePDG.h"
#include "TString.h"
#include "TF1.h"
#include "TRatioPlot.h"
#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
#include "TMath.h"
#include "TTree.h"
#include "TChain.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TSystem.h"
#include "TROOT.h"
#include "TGraph.h"
#include "TGraphErrors.h"
#include "TGraphAsymmErrors.h"
#include "TMultiGraph.h"
#include "TCanvas.h"
#include "TPad.h"
#include "TLegend.h"
#include "TLatex.h"
#include "TLine.h"
#include "TAxis.h"
#include "TGraph.h"
//#include "H1Boost.h"
//#include "H1PhysUtils/H1BoostedJets.h"
#include "TGraphErrors.h"
#include "TLorentzVector.h"
#include "TLorentzRotation.h"
#include "fastjet/PseudoJet.hh"
#include "fastjet/NNH.hh"
#include "LHAPDF/LHAPDF.h"
#include "CentauroPlugin.hh"
#include "CentauroPlugin.cc"

#include "Pythia8/Pythia.h"
//#include "HepMC/IO_GenEvent.h"
//#include "HepMC/GenEvent.h"
//using namespace HepMC;
using namespace Pythia8;
using namespace std;
using namespace LHAPDF;
using namespace fastjet;


int main( int argc, char* argv[]) {
  cout << "argc is " << argc << " argv[0] is " << argv[0] << " argv[1] is " << argv[1] << endl;
  gROOT->SetBatch(true);
  //defining histograms for multiplicity

  // Beam energies, minimal Q2, number of events to generate.
  int Q2min     = 150;
  int    nEvent    = 5000000; // dire: 10000 evt ~ 1min, 100000~4min
  //int    nEvent    = 100000; // dire: 10000 evt ~ 1min, 100000~4min
 
  // Generator. Shorthand for event.
  string xml = getenv("PYTHIA8__HOME");
  Pythia pythia(xml+"/share/Pythia8/xmldoc");
  Event& event = pythia.event;
  //double seed;
  // Rndm::init(seed);
  cout << "before pythia init" << endl;
  TFile* file1 = NULL;
  int random = rand() % 900000000+1;
  //cout << random << endl;
  pythia.readString("Random:setSeed  = on");
  pythia.readString(Form("Random:seed  = %i",random));
  pythia.readString("PDF:useHard  = off");
  pythia.readString("SpaceShower:alphaSvalue   = 0.118");
  pythia.readString("TimeShower:alphaSvalue    = 0.118");
  //pythia.readString("TimeShower:pTmin    = 0.1");
  //pythia.readString("SpaceShower:pTmin    = 0.1");
  //pythia.readString("PDF:pSet     = LHAPDF6:NNPDF31_nlo_as_0124"); 
  //pythia.readString("PDF:pHardSet = LHAPDF6:NNPDF31_nlo_as_0124");
  pythia.readString("PDF:pSet     = LHAPDF6:NNPDF31_nnlo_as_0118"); 
  pythia.readString("PDF:pHardSet = LHAPDF6:NNPDF31_nnlo_as_0118");
  pythia.readString("SigmaProcess:alphaSvalue = 0.118");
  const int ShowerModel = atoi(argv[2]);
   
  // ------------------------------------------------------------------------------ //
  if ( ShowerModel == 3 ) {
    // ------------------------------------------------------------------------------ //
    // for dire read:
    //  https://dire.gitlab.io
    //  https://gitlab.com/dire/direforpythia/-/blob/master/trunk/main/dis.cmnd
    //  https://dire.gitlab.io/Documentation/worksheet1500.pdf
    // ------------------------------------------------------------------------------ //
    //# Use external PDF sets from LHAPDF for incoming hadrons. 
    pythia.readString("PartonShowers:model = 3" ); //! 1:old showers, 2:vincia, 3:dire ! Warning: Dire comes with weighted events.
    pythia.readString("PDF:useHard  = on");
    pythia.readString("PDF:pSet = LHAPDF6:MMHT2014nlo68cl"); 
    pythia.readString("PDF:pHardSet = LHAPDF6:MMHT2014nlo68cl"); 
    pythia.readString("TimeShower:alphaSvalue     = 0.1201");
    pythia.readString("SpaceShower:alphaSvalue    = 0.1201");
    pythia.readString("PDF:pSet = LHAPDF6:MMHT2014nlo68cl"); 
    // pythia.readString("PDF:pHardSet = LHAPDF6:NNPDF31_nlo_as_0118");
    //# Use masses of PDF sets also for shower evolution. 
    //# Note: Only correctly handled when using external PDF sets from LHAPDF 
     
    pythia.readString("ShowerPDF:usePDFalphas    = on"); 
    pythia.readString("ShowerPDF:useSummedPDF    = on"); 
    pythia.readString("ShowerPDF:usePDFmasses    = off"); 
    pythia.readString("DireSpace:forceMassiveMap = on");    
    //# Disallow 'power shower'. 
    // pythia.readString("SpaceShower:pTmaxMatch    = 1"); // option 1 : Showers are always started at the factorisation scale.  
    // pythia.readString("TimeShower:pTmaxMatch     = 1"); // option 1 : Showers are always started at the factorisation scale. 
    // if(argv[1])
    //	{
    file1 = TFile::Open(Form("/nfs/dust/h1/group/vmikuni/jetdis/Pythia83/Extras/Dire/pythia83_dire_1M_%s.root",argv[1]),"RECREATE");
    //	}
    // else{
    //	file1 = TFile::Open("GIM_pythia83_dire_1M.root","RECREATE");
    //      }
    cout << "Dire file name is " << file1->GetName() << endl;
    // ------------------------------------------------------------------------------ //
  }
  else if  ( ShowerModel == 2 ) {
    pythia.readString("PartonShowers:model     = 2" ); //! 1:old showers, 2:vincia, 3:dire ! Warning: Dire comes with weighted events.
    pythia.readString("SpaceShower:pTmaxMatch  = 2" ); // option 2 : Showers are always started at the phase-space maximum. This option is not recommended for physics runs as it will lead to unphysical double counting in many cases. 
    pythia.readString("TimeShower:pTmaxMatch   = 1" ); // option 2 : Showers are always started at the phase-space maximum. This option is not recommended for physics runs as it will lead to 
    file1 = TFile::Open(Form("/nfs/dust/h1/group/vmikuni/jetdis/Pythia83/Extras/Vincia/pythia83_vincia_10M_118_%s.root",argv[1]),"RECREATE");
    // file1 = TFile::Open("GIM_pythia83_vincia_10M_118_Q_300.root","RECREATE");
  }
  else if  ( ShowerModel == 1 ) {
    pythia.readString("PartonShowers:model     = 1" ); //! 1:old showers, 2:vincia, 3:dire ! Warning: Dire comes with weighted events.
    pythia.readString("SpaceShower:pTmaxMatch  = 2" ); // option 2 : Showers are always started at the phase-space maximum. This option is not recommended for physics runs as it will lead to unphysical double counting in many cases. 
    pythia.readString("TimeShower:pTmaxMatch   = 1" ); // option 2 : Showers are always started at the phase-space maximum. This option is not recommended for physics runs as it will lead to 
    file1 = TFile::Open(Form("/nfs/dust/h1/group/vmikuni/jetdis/Pythia83/Extras/default/pythia83_default_10M_118_%s.root",argv[1]),"RECREATE");
    //      file1 = TFile::Open("tau1b_pythia83_default.root","RECREATE");
  }
   
  // Set up incoming beams, for frame with unequal beam energies.
  const double eProton   = 920.;
  const double eElectron = 27.6;
  pythia.readString("Beams:idA = 2212");  // BeamA = proton.
  pythia.settings.parm("Beams:eA", eProton);
  pythia.readString("Beams:idB = 11"); // BeamB = electron (-11) positron(11)
  pythia.settings.parm("Beams:eB", eElectron);
  pythia.readString("Beams:frameType = 2");

  //  
   
  // Set up DIS process within some phase space.
  pythia.readString(Form("PhaseSpace:Q2Min = %d",Q2min)); // Phase-space cut: minimal Q2 of process.
  pythia.readString("WeakBosonExchange:ff2ff(t:gmZ) = on");  // Neutral current (with gamma/Z interference).    
  pythia.readString("PDF:lepton = off"); // QED radiation off lepton not handled yet by the new procedure.
  pythia.readString("TimeShower:QEDshowerByL = off"); // QED radiation off lepton not handled yet by the new procedure.
  pythia.readString("SpaceShower:dipoleRecoil = on"); // Set dipole recoil on. Necessary for DIS + shower.
  //pythia.readString("SpaceShower:pTmaxMatch = 2"); // Allow emissions up to the kinematical limit, since rate known to match well to matrix elements everywhere.



  pythia.readString("HadronLevel:Hadronize = off"); // hadron-level on/off
  pythia.readString("PartonLevel:FSR = on");
  //pythia.readString("WeakBosonAndParton:all  = on");
  //pythia.readString("HardQCD:all = on");
  //pythia.readString("SoftQCD:all = on");

  // Uncomment to allow charged current.
  //pythia.readString("WeakBosonExchange:ff2ff(t:W) = on");

  // particle decays:
  pythia.readString("ParticleDecays:limitTau0 = on"); // When on, only particles with tau0 < tau0Max are decayed. 
  pythia.readString("ParticleDecays:tau0Max = 10");   //  The above tau0Max, expressed in mm/c. 
  // pythia.readString("ParticleDecays:limitTau = on");  // When on, only particles with tau < tauMax are decayed. 
  // pythia.readString("ParticleDecays:tauMax = 10");    //  The above tauMax, expressed in mm/c. 
  cout << "before pythia init" << endl;
  // Initialize.
  // pythia.settings.listAll();
  pythia.init();
  //pythia.settings.list("shower");
  // Histograms.
  cout << "before histograms" << endl;
  Double_t Q2_binning[5] {150.,360.42171212,866.02540378,2080.89572514,5000.};
  // TTree *tree = new TTree("Tree","Tree");
  Double_t eta;
  Double_t q2;
  // tree->Branch("eta",&eta,"eta/D");
  // tree->Branch("q2",&q2,"q2/D");

  //Q2 binned observables
  Double_t pt150_binning[7] {10, 16.050000953674196, 24.07000095367545, 32.80000095367653, 43.14000095367447, 57.160000953671684, 105.12264};
  Double_t pt237_binning[8] {10, 16.650001907348607, 26.420001907350134, 37.17000190734998, 51.74000190734708, 78.69000190735215, 83.03000190735438, 100.90953};
  Double_t pt346_binning[9] {10, 21.190002861023633, 36.510002861024425, 58.74000286102, 71.57000286102283, 84.46000286102942, 99.16000286103694, 117.95000286104656, 143.86293};
  Double_t eta150_binning[7] {-1, -0.5218995231628945, -0.2855995231629205, -0.042099523162935776, 0.2636004768370517, 0.7280004768370005, 2.4999971};
  Double_t eta237_binning[9] {-1, -0.4987999403954104, -0.2892999403954335, -0.10419994039545098, 0.07640005960455212, 0.2666000596045379, 0.46590005960451597, 0.7634000596044832, 2.4999995};
  Float_t eta346_binning[] = {-1,-0.11549982118616176, 0.1602001788138392, 0.40660017881381205, 0.6466001788137856, 1.0048001788137462, 2.4999936};
  Double_t phi150_binning[7] {0, 0.14950005215406403, 0.558100052154019, 0.8652000521539852, 1.1659000521539522, 1.5232000521539129, 1.570620059967041};
  Double_t phi237_binning[9] {0, 0.09770005960464655, 0.3703000596046203, 0.5800000596045972, 0.8009000596045729, 1.01040005960455, 1.2351000596045252, 1.473000059604499, 1.5705121755599976};
  Double_t phi346_binning[9] {0.0, 0.06160000000000074, 0.3050999999999827, 0.5131999999999598, 0.728899999999936, 0.9413999999999126, 1.156699999999889, 1.3817999999998642, 1.5706232786178589};
  Double_t qt150_binning[8] {0, 0.2626663129903716, 0.7018663129903232, 1.249766312990263, 1.8557663129901962, 2.745866312991754, 4.381366312993512, 8.171009};
  Double_t qt237_binning[9] {0, 0.2774143999677533, 0.6396143999677134, 1.071414399967666, 1.6017143999676076, 2.2541143999681, 3.6533143999710527, 5.329314399968686, 6.249228};
  Double_t qt346_binning[6] {0, 0.3009084919101769, 0.7532084919101271, 1.2760084919100696, 2.0770084919101524, 5.054179};

  //y binned observables
  Double_t pt2_binning[7] {10.000006, 18.890005722046222, 32.90000572204809, 71.07000572204552, 84.75000572205252, 112.21000572206657, 124.81343};
  Double_t pt31_binning[6] {10.000003, 24.090002861024086, 75.56000286102487, 91.68000286103312, 109.07000286104201, 139.06075};
  Double_t pt44_binning[11] {10.000004, 26.590003814698793, 46.400003814696774, 57.88000381469449, 69.48000381469608, 81.47000381470221, 92.3900038147078, 104.41000381471395, 118.09000381472094, 131.91000381472244, 143.03601};
  Double_t eta2_binning[7] {-0.5050307, -0.07563069162372818, 0.23276930837626364, 0.5192693083762321, 0.8820693083761921, 1.5605693083761174, 2.499989};
  Double_t eta31_binning[6] {-0.8509098, -0.42630982913975624, -0.1543098291397862, 0.1611901708602117, 0.6254901708601606, 2.4999971};
  Double_t eta44_binning[6] {-0.9999995, -0.6128995231628844, -0.3632995231629119, -0.09299952316293723, 0.5558004768370195, 2.4999936};
  Double_t phi2_binning[7] {2.9802322e-08, 0.1026000298023243, 0.5320000298022801, 0.8518000298022449, 1.152100029802212, 1.4954000298021741, 1.5705938339233398};
  Double_t phi31_binning[7] {0.0, 0.2017999999999941, 0.6337999999999465, 0.9443999999999123, 1.2432999999998795, 1.5619999999998444, 1.570620059967041};
  Double_t phi44_binning[6] {5.9604645e-08, 0.44800005960461176, 0.8419000596045684, 1.1664000596045327, 1.5161000596044942, 1.570563793182373};
  Double_t qt2_binning[9] {4.7180823e-05, 0.23574718082344825, 0.5346471808234153, 0.8288471808233829, 1.183647180823344, 1.6080471808232972, 2.290547180823867, 3.4194471808262494, 4.967163};
  Double_t qt31_binning[7] {4.4488774e-05, 0.3935444887736833, 0.9438444887736227, 1.5278444887735585, 2.363744488774274, 3.621344488776928, 5.4841785};
  Double_t qt44_binning[9] {1.732245e-05, 0.2536173224507191, 0.6306173224506776, 1.0896173224506271, 1.645517322450566, 2.33801732245124, 3.161017322452977, 4.18321732245432, 6.2783265};
      
  

  TH1D* Q2_hist = new TH1D("Q2hist","Q2",4,Q2_binning);
  TH1D* ncharge_hist= new TH1D("gen_jet_ncharged","gen_jet_ncharged",19,1,20);
  TH1D* charge_hist= new TH1D("gen_jet_charge","gen_jet_charge",9,-0.8,0.8);
  TH1D* ptD_hist = new TH1D("gen_jet_ptD","gen_jet_ptD",9,0.3,0.7);
  TH1D* tau10_hist = new TH1D("gen_jet_tau10","gen_jet_tau10",7,-2.2,-1);
  TH1D* tau15_hist= new TH1D("gen_jet_tau15","gen_jet_tau15",7,-3.0,-1.2);
  TH1D* tau20_hist= new TH1D("gen_jet_tau20","gen_jet_tau20",7,-3.5,-1.5);
  TH1D* z_hist= new TH1D("gen_jet_z","gen_jet_z",7,0.1,1);
  TH1D* z1_hist= new TH1D("gen_jet_z1","gen_jet_z1",7,0.1,1);
  
  TH2D* ncharge_hist2D= new TH2D("gen_jet_ncharged2D","gen_jet_ncharged2D",19,1,20,4,Q2_binning);
  TH2D* charge_hist2D= new TH2D("gen_jet_charge2D","gen_jet_charge2D",9,-0.8,0.8,4,Q2_binning);
  TH2D* ptD_hist2D = new TH2D("gen_jet_ptD2D","gen_jet_ptD2D",9,0.3,0.7,4,Q2_binning);
  TH2D* tau10_hist2D = new TH2D("gen_jet_tau102D","gen_jet_tau102D",7,-2.2,-1,4,Q2_binning);
  TH2D* tau15_hist2D= new TH2D("gen_jet_tau152D","gen_jet_tau152D",7,-3.0,-1.2,4,Q2_binning);
  TH2D* tau20_hist2D= new TH2D("gen_jet_tau202D","gen_jet_tau202D",7,-3.5,-1.5,4,Q2_binning);
  TH2D* z_hist2D= new TH2D("gen_jet_z2D","gen_jet_z2D",7,0.1,1,4,Q2_binning);
  TH2D* z1_hist2D= new TH2D("gen_jet_z12D","gen_jet_z12D",7,0.1,1,4,Q2_binning);
  

  TH1D* Q2_150_jetpt = new TH1D("Q2_150_jetpt","Q2_150_jetpt",6,pt150_binning);
  TH1D* Q2_150_jeteta = new TH1D("Q2_150_jeteta","Q2_150_jeteta",6,eta150_binning);
  //TH1D* Q2_150_jeteta = new TH1D("Q2_150_jeteta","Q2_150_jeteta",5,-1,2.5);
  TH1D* Q2_150_jetdphi = new TH1D("Q2_150_jetdphi","Q2_150_jetdphi",6,phi150_binning);
  TH1D* Q2_150_jetqt = new TH1D("Q2_150_qt","Q2_150_qt",7,qt150_binning);

  TH1D* Q2_237_jetpt = new TH1D("Q2_237_jetpt","Q2_237_jetpt",7,pt237_binning);
  TH1D* Q2_237_jeteta = new TH1D("Q2_237_jeteta","Q2_237_jeteta",8,eta237_binning);
  TH1D* Q2_237_jetdphi = new TH1D("Q2_237_jetdphi","Q2_237_jetdphi",8,phi237_binning);
  TH1D* Q2_237_jetqt = new TH1D("Q2_237_qt","Q2_237_qt",8,qt237_binning);

  TH1D* Q2_346_jetpt = new TH1D("Q2_346_jetpt","Q2_346_jetpt",8,pt346_binning);
  Int_t  binnum = sizeof(eta346_binning)/sizeof(Float_t) - 1;
  TH1F* Q2_346_jeteta = new TH1F("Q2_346_jeteta","Q2_346_jeteta",binnum,eta346_binning);

  //TH1D* Q2_346_jeteta = new TH1D("Q2_346_jeteta","Q2_346_jeteta",10,-1,2.5);
  TH1D* Q2_346_jetdphi = new TH1D("Q2_346_jetdphi","Q2_346_jetdphi",8,phi346_binning);
  TH1D* Q2_346_jetqt = new TH1D("Q2_346_qt","Q2_346_qt",5,qt346_binning);


  TH1D* y_2_jetpt = new TH1D("y_2_jetpt","y_2_jetpt",6,pt2_binning);
  TH1D* y_2_jeteta = new TH1D("y_2_jeteta","y_2_jeteta",6,eta2_binning);
  TH1D* y_2_jetdphi = new TH1D("y_2_jetdphi","y_2_jetdphi",6,phi2_binning);
  TH1D* y_2_jetqt = new TH1D("y_2_qt","y_2_qt",8,qt2_binning);

  TH1D* y_31_jetpt = new TH1D("y_31_jetpt","y_31_jetpt",5,pt31_binning);
  TH1D* y_31_jeteta = new TH1D("y_31_jeteta","y_31_jeteta",5,eta31_binning);
  TH1D* y_31_jetdphi = new TH1D("y_31_jetdphi","y_31_jetdphi",6,phi31_binning);
  TH1D* y_31_jetqt = new TH1D("y_31_qt","y_31_qt",6,qt31_binning);

  TH1D* y_44_jetpt = new TH1D("y_44_jetpt","y_44_jetpt",10,pt44_binning);
  TH1D* y_44_jeteta = new TH1D("y_44_jeteta","y_44_jeteta",5,eta44_binning);
  TH1D* y_44_jetdphi = new TH1D("y_44_jetdphi","y_44_jetdphi",5,phi44_binning);
  TH1D* y_44_jetqt = new TH1D("y_44_qt","y_44_qt",8,qt44_binning);


   
  // pythia histograms
  Hist dtau1b ("Jet pt cross-section", 20, 0, 1.0);
  Hist dtot   ("total cross-section", 1, 0, 1.);
  TH1D htot   ("total cross-section","total cross-section",1,0,1);
  cout << "after histograms" << endl;
	
  // Four-momenta of proton, electron, virtual photon/Z^0/W^+-.
       
  // Begin event loop.
  //cout << "made it to event loop" << endl;
  for (int iEvent = 0; iEvent < nEvent; ++iEvent) {
    if (!pythia.next()) continue;
	  
    TLorentzVector part4v;
    //TLorentzVector HFS_sum;
    //cout << "event size (including mothers) is " << event.size() << endl;
    // Four-momenta of proton, electron, virtual photon/Z^0/W^+-.                                                                                                                                       
    Vec4 pProton = event[1].p();
    Vec4 peIn    = event[4].p();
    int pdgidlepton = event[4].id();
    int idelec = 4;
    for ( ; idelec<100 ; idelec++  ) {
      if ( event[idelec].id() == pdgidlepton && event[idelec].isFinal() ) break;
    }
    Vec4 peOut   = event[idelec].p();
    Vec4 pPhoton = peIn - peOut;

    TLorentzVector ebeam(peIn.px(),peIn.py(),peIn.pz(),peIn.e());
    TLorentzVector pbeam(pProton.px(),pProton.py(),pProton.pz(),pProton.e());
    TLorentzVector scat_e(peOut.px(),peOut.py(),peOut.pz(),peOut.e());
    double Q2    = - pPhoton.m2Calc();
    TLorentzVector Elec(peOut.px(), peOut.py(), peOut.pz(), peOut.e()); // scattered electron    
    // TLorentzVector pPhoton = peIn - Elec;

    double X     = Q2 / (2. * pProton * pPhoton);
    double Y     = (pProton * pPhoton) / (pProton * peIn);
	
    dtot.fill(0.5);
    htot.Fill(0.5,1);
        
    // Fill kinematics histograms.
    Q2_hist->Fill( Q2 );

        
    TLorentzVector part4vStar;
    TLorentzVector Photon(pPhoton.px(),pPhoton.py(),pPhoton.pz(),pPhoton.e());
    TLorentzVector peInTL(peIn.px(),peIn.py(),peIn.pz(),peIn.e());
    TLorentzVector qJ = peInTL - Elec + X*pbeam;
    TLorentzVector qB = X*pbeam;

    double weight = pythia.info.weight();

    // dire! veto events with large, or 0, weight
    if (abs(weight) > 1e3) {     
      cout << "Warning in DIRE main program dire03.cc: Large shower weight wt="       << weight << endl; 
      if (abs(weight) > 1e4) {     
	cout << "Warning in DIRE main program dire03.cc: Shower weight larger"         << " than 10000. Discard event with rare shower weight fluctuation."         << endl; 
	weight = 0.;  
      }  
    }     
    // Do not print zero-weight events.  
    if ( weight == 0. ) continue;

    vector<PseudoJet> particlelist;
    TLorentzVector jet_vec;
    //cout << "event size (including mothers) is " << event.size() << endl;
    for (int i = 0; i < event.size(); ++i) {  // we start from: i>6
      if (event[i].isFinal()){
	if (event[i].idAbs() == 12 || event[i].idAbs() == 14 ||
	    event[i].idAbs() == 16)     continue;
	if(pythia.event[i].idAbs() ==11) continue;
	if ( !event[i].isVisible()  ) continue;
	if ( event[i].mother1()==6 ) continue; //remove scattered lepton 
	Vec4 part4vect = event[i].p();
	fastjet::PseudoJet gen_particle(part4vect.px(),part4vect.py(),part4vect.pz(),part4vect.e());
	gen_particle.set_user_index(event[i].charge());      
	part4v.SetPxPyPzE(part4vect.px(),part4vect.py(),part4vect.pz(),part4vect.e());
	if (part4v.Eta()<5 && part4v.Pt()>0.1){
	  particlelist.push_back(gen_particle);
	}

      }

	
    }

    // End of particle loop.
    if (( Y > 0.2 && Y<0.7) && peOut.e() > 11  && Q2>150) {
      fastjet::RecombinationScheme    recombScheme = fastjet::E_scheme;
      fastjet::Strategy               strategy = fastjet::Best;
      fastjet::JetDefinition jet_def(fastjet::kt_algorithm, 1.0,recombScheme, strategy);
      
      ClusterSequence clust_seq_lab(particlelist, jet_def);
      vector<PseudoJet> jets_lab = clust_seq_lab.inclusive_jets(10.0);
      vector<PseudoJet> sortedLabJets = sorted_by_pt(jets_lab);
      for (unsigned ijet= 0; ijet < sortedLabJets.size();ijet++) {
	fastjet::PseudoJet jet = sortedLabJets[ijet];
	if (jet.eta() > 2.5 || jet.eta() < -1) continue;
	jet_vec.SetPtEtaPhiE(jet.perp(), jet.eta(), jet.phi(), jet.e());
	vector<PseudoJet> constituents = jet.constituents();
	int ncharged = 0;
	float jet_charge = 0;
	float jet_ptD = 0;
	float tau10 = 0;
	float tau15 = 0;
	float tau20 = 0;
	float sumpt=0;
	float jet_abs = pow(jet.px(), 2) + pow(jet.py(), 2)+pow(jet.pz(), 2);
	float z=0;
	float z1=0;
	float maxpt=0;
	float qt = sqrt( (jet.px() + peOut.px() )*(jet.px() + peOut.px() )  + (jet.py() + peOut.py() )*(jet.py() + peOut.py() ));
	float dphi = 3.14159265359 - jet_vec.DeltaPhi(scat_e);
	//cout << jet.eta() << endl;
	if (Q2>150 && Q2<237){
	  Q2_150_jetpt->Fill(jet.perp());
	  Q2_150_jeteta->Fill(jet.eta());
	  Q2_150_jetdphi->Fill(dphi);
	  Q2_150_jetqt->Fill(qt/sqrt(Q2));
	}
	else if (Q2>237 && Q2<346.5){
	  Q2_237_jetpt->Fill(jet.perp());
	  Q2_237_jeteta->Fill(jet.eta());
	  Q2_237_jetdphi->Fill(dphi);
	  Q2_237_jetqt->Fill(qt/sqrt(Q2));
	}
	else if (Q2>346.5 && Q2<51932.){
	  Q2_346_jetpt->Fill(jet.perp());
	  Q2_346_jeteta->Fill(jet.eta());
	  Q2_346_jetdphi->Fill(dphi);
	  Q2_346_jetqt->Fill(qt/sqrt(Q2));	  
	  q2 = Q2;
	  eta = jet.eta();
	  //tree->Fill();
	}

	if (Y>0.2 && Y<0.31){
	  y_2_jetpt->Fill(jet.perp());
	  y_2_jeteta->Fill(jet.eta());
	  y_2_jetdphi->Fill(dphi);
	  y_2_jetqt->Fill(qt/sqrt(Q2));
	}
	else if (Y>0.31 && Y<0.44){
	  y_31_jetpt->Fill(jet.perp());
	  y_31_jeteta->Fill(jet.eta());
	  y_31_jetdphi->Fill(dphi);
	  y_31_jetqt->Fill(qt/sqrt(Q2));
	}
	else if (Y>0.44 && Y<0.7){
	  y_44_jetpt->Fill(jet.perp());
	  y_44_jeteta->Fill(jet.eta());
	  y_44_jetdphi->Fill(dphi);
	  y_44_jetqt->Fill(qt/sqrt(Q2));
	}
	for (unsigned j = 0; j < constituents.size(); j++) {
	  TVector3 hvector(constituents[j].px(), constituents[j].py(), constituents[j].pz());
	  TLorentzVector constituent(hvector,0);
	  jet_ptD += pow( hvector.Perp(), 2); 
	  sumpt+=hvector.Perp();
	  tau10+=hvector.Perp()*pow(constituent.DeltaR(jet_vec),1);
	  tau15+=hvector.Perp()*pow(constituent.DeltaR(jet_vec),1.5);
	  tau20+=hvector.Perp()*pow(constituent.DeltaR(jet_vec),2);

	  z = (constituent.Px()*jet.px() + constituent.Py()*jet.py() + constituent.Pz()*jet.pz())/jet_abs;
	  z_hist->Fill(z,weight);
	  z_hist2D->Fill(z,Q2,weight);
	  if (constituent.Pt()>maxpt){
	    maxpt = constituent.Pt();
	    z1=z;
	  }

	  if(!(constituents[j].user_index()!=0)) continue;
	  ncharged = ncharged +1;
	  jet_charge += constituents[j].user_index() *hvector.Perp(); 
	  
	}// end loop over jet constituents
	ncharge_hist->Fill(ncharged,weight);
	//cout << ncharged << endl;
	charge_hist->Fill(jet_charge/jet.perp(),weight);
	ptD_hist->Fill(pow(jet_ptD,0.5)/sumpt,weight);
	tau10_hist->Fill(TMath::Log(tau10/jet.perp()),weight);
	tau15_hist->Fill(TMath::Log(tau15/jet.perp()),weight);
	tau20_hist->Fill(TMath::Log(tau20/jet.perp()),weight);
	z1_hist->Fill(z1,weight);
	
	ncharge_hist2D->Fill(ncharged,Q2,weight);
	charge_hist2D->Fill(jet_charge/jet.perp(),Q2,weight);
	ptD_hist2D->Fill(pow(jet_ptD,0.5)/sumpt,Q2,weight);
	tau10_hist2D->Fill(TMath::Log(tau10/jet.perp()),Q2,weight);
	tau15_hist2D->Fill(TMath::Log(tau15/jet.perp()),Q2,weight);
	tau20_hist2D->Fill(TMath::Log(tau20/jet.perp()),Q2,weight);
	z1_hist2D->Fill(z1,Q2,weight);

      }
    }
  
	
  }// End of event loop. Statistics and histograms.
  cout << "after event loop " << endl;
  pythia.stat();    
  double sigmaGen = pythia.info.sigmaGen();
  sigmaGen *= 1.e9;// mb -> pb
  double xserr = pythia.info.sigmaErr();
  //double ntrials = pythia.info.nTried();
  double naccptd = pythia.info.nAccepted();
  //double nselctd = pythia.info.nSelected();
  double lumi = naccptd/sigmaGen;

  //cout<<"ntrials: "<<ntrials<<endl;
  cout<<"xsec: "<<sigmaGen<<endl;
  cout<<"xserr: "<<xserr<<endl;
  cout<<"lumi:  "<<lumi<<endl;

  double xstotPS = htot.GetBinContent(1);
  cout<<"Total number of events in PS: "<<xstotPS<<endl;
  xstotPS /= lumi;
  cout<<"PS/nonPS: "<< htot.GetBinContent(1)  / naccptd  <<endl;
  cout<<"Total cross section:   "<<xstotPS<<endl;
  cout<<"Total cross section 2: "<< htot.GetBinContent(1)  / naccptd * sigmaGen <<endl;

  //tree->Write();
  Q2_hist->Write();
  ncharge_hist->Write();
  charge_hist->Write();
  ptD_hist->Write();
  tau10_hist->Write();
  tau15_hist->Write();
  tau20_hist->Write();
  z_hist->Write();
  z1_hist->Write();
  
  ncharge_hist2D->Write();
  charge_hist2D->Write();
  ptD_hist2D->Write();
  tau10_hist2D->Write();
  tau15_hist2D->Write();
  tau20_hist2D->Write();
  z_hist2D->Write();
  z1_hist2D->Write();



  Q2_150_jetpt->Scale(1.0/Q2_150_jetpt->Integral(),"width");
  Q2_150_jetpt->Write();
  Q2_150_jeteta->Scale(1.0/Q2_150_jeteta->Integral(),"width");
  Q2_150_jeteta->Write();
  Q2_150_jetdphi->Scale(1.0/Q2_150_jetdphi->Integral(),"width");
  Q2_150_jetdphi->Write();
  Q2_150_jetqt->Scale(1.0/Q2_150_jetqt->Integral(),"width");
  Q2_150_jetqt->Write();

  Q2_237_jetpt->Scale(1.0/Q2_237_jetpt->Integral(),"width");
  Q2_237_jetpt->Write();
  Q2_237_jeteta->Scale(1.0/Q2_237_jeteta->Integral(),"width");
  Q2_237_jeteta->Write();
  Q2_237_jetdphi->Scale(1.0/Q2_237_jetdphi->Integral(),"width");
  Q2_237_jetdphi->Write();
  Q2_237_jetqt->Scale(1.0/Q2_237_jetqt->Integral(),"width");
  Q2_237_jetqt->Write();

  Q2_346_jetpt->Scale(1.0/Q2_346_jetpt->Integral(),"width");
  Q2_346_jetpt->Write();
  Q2_346_jeteta->Scale(1.0/Q2_346_jeteta->Integral(),"width");
  Q2_346_jeteta->Write();
  Q2_346_jetdphi->Scale(1.0/Q2_346_jetdphi->Integral(),"width");
  Q2_346_jetdphi->Write();
  Q2_346_jetqt->Scale(1.0/Q2_346_jetqt->Integral(),"width");
  Q2_346_jetqt->Write();

  y_2_jetpt->Scale(1.0/y_2_jetpt->Integral(),"width");
  y_2_jetpt->Write();
  y_2_jeteta->Scale(1.0/y_2_jeteta->Integral(),"width");
  y_2_jeteta->Write();
  y_2_jetdphi->Scale(1.0/y_2_jetdphi->Integral(),"width");
  y_2_jetdphi->Write();
  y_2_jetqt->Scale(1.0/y_2_jetqt->Integral(),"width");
  y_2_jetqt->Write();

  y_31_jetpt->Scale(1.0/y_31_jetpt->Integral(),"width");
  y_31_jetpt->Write();
  y_31_jeteta->Scale(1.0/y_31_jeteta->Integral(),"width");
  y_31_jeteta->Write();
  y_31_jetdphi->Scale(1.0/y_31_jetdphi->Integral(),"width");
  y_31_jetdphi->Write();
  y_31_jetqt->Scale(1.0/y_31_jetqt->Integral(),"width");
  y_31_jetqt->Write();

  y_44_jetpt->Scale(1.0/y_44_jetpt->Integral(),"width");
  y_44_jetpt->Write();
  y_44_jeteta->Scale(1.0/y_44_jeteta->Integral(),"width");
  y_44_jeteta->Write();
  y_44_jetdphi->Scale(1.0/y_44_jetdphi->Integral(),"width");
  y_44_jetdphi->Write();
  y_44_jetqt->Scale(1.0/y_44_jetqt->Integral(),"width");
  y_44_jetqt->Write();








  file1->Close();
  return 0;
}
