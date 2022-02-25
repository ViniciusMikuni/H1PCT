// -*- C++ -*-
#include "Rivet/Analysis.hh"
#include "Rivet/Projections/FinalState.hh"
#include "Rivet/Projections/FastJets.hh"
#include "Rivet/Projections/DISKinematics.hh"
#include "Rivet/Projections/DISFinalState.hh"

namespace Rivet {


  /// @brief Add a short analysis description here
  class H1_2022_TMD : public Analysis {
  public:

    /// Constructor
    //DEFAULT_RIVET_ANALYSIS_CTOR(H1_2022_TMD);
    RIVET_DEFAULT_ANALYSIS_CTOR(H1_2022_TMD);

    /// @name Analysis methods
    ///@{

    /// Book histograms and initialise projections before the run
    void init() {

      // Initialise and register projections

      DISLepton lepton(options());
      declare(lepton, "Lepton");
      declare(DISKinematics(lepton), "Kinematics");
      declare(FinalState(), "FS");
      const DISFinalState& disfs = declare(DISFinalState(DISFinalState::BoostFrame::LAB), "DISFS");

      FastJets jetfs(disfs, FastJets::KT, 1.0, JetAlg::Muons::NONE, JetAlg::Invisibles::NONE);
      declare(jetfs, "jets");

      // Book histograms
      
      Histo1DPtr tmp;
      _h_Q2_jetpt.add( 150.,  237.,  book(tmp, 1,1,1));
      _h_Q2_jetpt.add( 237.,346.5 ,  book(tmp, 2,1,1));
      _h_Q2_jetpt.add( 346.5,51932., book(tmp, 3,1,1));
     
      _h_Q2_jeteta.add( 150.,  237.,  book(tmp, 4,1,1));
      _h_Q2_jeteta.add( 237.,346.5 ,  book(tmp, 5,1,1));
      _h_Q2_jeteta.add( 346.5,51932., book(tmp, 6,1,1));
//
      _h_Q2_dphi.add( 150.,  237.,  book(tmp, 7,1,1));
      _h_Q2_dphi.add( 237.,346.5 ,  book(tmp, 8,1,1));
      _h_Q2_dphi.add( 346.5,51932., book(tmp, 9,1,1));

      _h_Q2_qt.add( 150.,  237.,  book(tmp, 10,1,1));
      _h_Q2_qt.add( 237.,346.5 ,  book(tmp, 11,1,1)); 
      _h_Q2_qt.add( 346.5,51932.,book(tmp, 12,1,1));
      
      _h_y_jetpt.add( 0.2,  0.31,  book(tmp, 13,1,1));
      _h_y_jetpt.add( 0.31, 0.44,  book(tmp, 14,1,1));
      _h_y_jetpt.add( 0.44, 0.7 ,  book(tmp, 15,1,1));
      
      _h_y_jeteta.add( 0.2,  0.31,  book(tmp, 16,1,1));
      _h_y_jeteta.add( 0.31, 0.44,  book(tmp, 17,1,1));
      _h_y_jeteta.add( 0.44, 0.7 ,  book(tmp, 18,1,1));
      
      _h_y_dphi.add( 0.2,  0.31,  book(tmp, 19,1,1));
      _h_y_dphi.add( 0.31, 0.44,  book(tmp, 20,1,1));
      _h_y_dphi.add( 0.44, 0.7 ,  book(tmp, 21,1,1));
      
      _h_y_qt.add( 0.2,  0.31,  book(tmp, 22,1,1));
      _h_y_qt.add( 0.31, 0.44,  book(tmp, 23,1,1));
      _h_y_qt.add( 0.44, 0.7 ,  book(tmp, 24,1,1));

    }


    /// Perform the per-event analysis
    void analyze(const Event& event) {

      // Get the DIS kinematics
      const DISKinematics& dk = apply<DISKinematics>(event, "Kinematics");
      if ( dk.failed() ) return;
      // Momentum of the scattered lepton
      const DISLepton& dl = apply<DISLepton>(event,"Lepton");
      if ( dl.failed() ) return;

      FourMomentum eout = dl.out().momentum()*GeV; //Weird unit mixing, dl.out gives units of GeV but dl.in() gives MeV??? todo                                                          
      FourMomentum ein = dl.in().momentum()*MeV;
      FourMomentum proton = dk.beamHadron().momentum()*MeV;
      FourMomentum photon = ein-eout;
      
      double Q2 = -photon.mass2();
      //double x     = Q2 / (2.*proton*photon);
      double y     = (proton * photon) / (proton* ein);

      // double x  = dk.x();
      // double y  = dk.y();
      // double Q2 = dk.Q2();

      if (Q2 < 150.0*GeV2) vetoEvent;
      if ( y>0.7) vetoEvent;
      if(y<0.2) vetoEvent;
      
      // Weight of the event
      //_hist_Q2->fill(Q2);
      //_hist_y->fill(y);
      //_hist_x->fill(x);

      const FourMomentum leptonMom = dl.out();
      //const double enel = leptonMom.E();
      const double enel = eout.E();
      
      if(enel<11*GeV) vetoEvent;
      
      // Retrieve clustered jets, sorted by pT, with a minimum pT cut
      // float qt = 0;
      Jets jets = apply<FastJets>(event, "jets").jetsByPt(Cuts::pT > 10*GeV && Cuts::eta < 2.5 && Cuts::eta>-1.0);  
      int njets = jets.size(); 
      for (int i = 0; i < njets; ++i){
	  FourMomentum jetmom = jets[i].momentum();
        //_h["jetpt"]->fill(jets[i].pT()/GeV);
	  //_h["jeteta"]->fill(jets[i].eta());
	  float qt = sqrt( (jetmom.px() + leptonMom.px() )/GeV*(jetmom.px() + leptonMom.px() )/GeV  + (jetmom.py() + leptonMom.py() )/GeV*(jetmom.py() + leptonMom.py() )/GeV           )           ;
	  //_h["qt"]->fill(qt/sqrt(Q2));
          float dphi = 3.14159265359- mapAngle0ToPi(leptonMom.phi()-jets[i].phi());
	  //_h["dphi"]->fill(dphi);
       
        _h_Q2_jetpt.fill(Q2,jets[i].pT()/GeV);
        _h_Q2_jeteta.fill(Q2,jets[i].eta());
        _h_Q2_qt.fill(Q2,qt/sqrt(Q2));
        _h_Q2_dphi.fill(Q2,dphi);

        _h_y_jetpt.fill(y,jets[i].pT()/GeV);
        _h_y_jeteta.fill(y,jets[i].eta());
        _h_y_qt.fill(y,qt/sqrt(Q2));
        _h_y_dphi.fill(y,dphi);

      }


    }


    /// Normalise histograms etc., after the run
    void finalize() {

      //normalize(_h["dphi"]); // normalize to unity
      //normalize(_h["qt"]); // normalize to unity
      //normalize(_hist_y); // normalize to unity
      //normalize(_hist_x); // normalize to unity
      //normalize(_h["jetpt"]); // normalize to unity
      //normalize(_h["jeteta"]); // normalize to unity
      
      for (Histo1DPtr histo : _h_Q2_jetpt.histos()) normalize(histo);
      for (Histo1DPtr histo : _h_Q2_jeteta.histos()) normalize(histo);
      for (Histo1DPtr histo : _h_Q2_qt.histos()) normalize(histo);
      for (Histo1DPtr histo : _h_Q2_dphi.histos()) normalize(histo);

      for (Histo1DPtr histo : _h_y_jetpt.histos()) normalize(histo);
      for (Histo1DPtr histo : _h_y_jeteta.histos()) normalize(histo);
      for (Histo1DPtr histo : _h_y_qt.histos()) normalize(histo);
      for (Histo1DPtr histo : _h_y_dphi.histos()) normalize(histo);

    }

    ///@}


    /// @name Histograms
    ///@{
    Histo1DPtr _hist_Q2, _hist_y, _hist_x, _hist_ept;
    map<string, Histo1DPtr> _h;
    map<string, Profile1DPtr> _p;
    map<string, CounterPtr> _c;
    
    BinnedHistogram _h_Q2_jetpt,_h_Q2_jeteta, _h_Q2_dphi, _h_Q2_qt ;
    BinnedHistogram _h_y_jetpt,_h_y_jeteta, _h_y_dphi, _h_y_qt ;
    ///@}


  };


  //DECLARE_RIVET_PLUGIN(H1_2022_TMD);
  RIVET_DECLARE_PLUGIN(H1_2022_TMD);
}
