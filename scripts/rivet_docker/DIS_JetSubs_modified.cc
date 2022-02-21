#include "Rivet/Analysis.hh"
#include "Rivet/Projections/FinalState.hh"
#include "Rivet/Projections/FastJets.hh"
#include "Rivet/Projections/DISKinematics.hh"
#include "Rivet/Projections/DISFinalState.hh"
namespace Rivet {
  /// @brief Add a short analysis description here
  class DIS_JetSubs : public Analysis {
  public:
    /// Constructor
    DEFAULT_RIVET_ANALYSIS_CTOR(DIS_JetSubs);
    /// Book histograms and initialise projections before the run
    void init() {

      // Initialise and register projections. Note that the definition
      // of the scattered lepton can be influenced by sepcifying
      // options as declared in the .info file.
      DISLepton lepton(options());
      declare(lepton, "Lepton");
      declare(DISKinematics(lepton), "Kinematics");
      declare(FinalState(), "FS");
      const DISFinalState& disfs = declare(DISFinalState(DISFinalState::BoostFrame::LAB), "DISFS");

      FastJets jetfs(disfs, FastJets::KT, 1.0, JetAlg::Muons::NONE, JetAlg::Invisibles::NONE);
      declare(jetfs, "jets");

      // Book histograms
      
      book(_hist_Q2, "Q2",logspace(4,150, 5000.0));
      
      book(_hist_ncharge, "gen_jet_ncharged",linspace(19,1,20));
      book(_hist_charge, "gen_jet_charge",9,-0.8,0.8);
      book(_hist_ptD, "gen_jet_ptD",9,0.3,0.7);
      book(_hist_tau10, "gen_jet_tau10",7,-2.2,-1);
      book(_hist_tau15, "gen_jet_tau15",7,-3.0,-1.2);
      book(_hist_tau20, "gen_jet_tau20",7,-3.5,-1.5);

      book(_hist_ncharge2D, "gen_jet_ncharged2D",linspace(19,1,20,5),logspace(4,150, 5000.0));
      book(_hist_charge2D, "gen_jet_charge2D",linspace(9,-0.8,0.8),logspace(4,150, 5000.0));
      book(_hist_ptD2D, "gen_jet_ptD2D",linspace(9,0.3,0.7),logspace(4,150, 5000.0));
      book(_hist_tau102D, "gen_jet_tau102D",linspace(7,-2.2,-1),logspace(4,150, 5000.0));
      book(_hist_tau152D, "gen_jet_tau152D",linspace(7,-3.0,-1.2),logspace(4,150, 5000.0));
      book(_hist_tau202D, "gen_jet_tau202D",linspace(7,-3.5,-1.5),logspace(4,150, 5000.0));
      
    }


    /// Perform the per-event analysis
    void analyze(const Event& event) {
      // Get the DIS kinematics
      const DISKinematics& dk = apply<DISKinematics>(event, "Kinematics");
      if ( dk.failed() ) return;
      const DISLepton& dl = apply<DISLepton>(event,"Lepton");
      if ( dl.failed() ) return;
      
      FourMomentum eout = dl.out().momentum()*GeV; //Weird unit mixing, dl.out gives units of GeV but dl.in() gives MeV??? todo                                                          
      FourMomentum ein = dl.in().momentum()*MeV;
      FourMomentum proton = dk.beamHadron().momentum()*MeV;
      FourMomentum photon = ein-eout;
      
      double Q2 = -photon.mass2();
      //double x     = Q2 / (2.*proton*photon);
      double y     = (proton * photon) / (proton* ein);
     
      if (Q2 < 150.0*GeV2) vetoEvent;
      if (y>0.7) vetoEvent;
      if(y<0.2) vetoEvent;
            // Weight of the event
      _hist_Q2->fill(Q2);

      // Momentum of the scattered lepton
      const double enel = eout.E();
      if(enel<11*GeV) vetoEvent;
      // Extract the particles other than the lepton
      const FinalState& fs = apply<FinalState>(event, "FS");
      Particles particles;
      particles.reserve(fs.particles().size());
      ConstGenParticlePtr dislepGP = dl.out().genParticle();
      for(const Particle& p: fs.particles()) {
        ConstGenParticlePtr loopGP = p.genParticle();
        if (loopGP == dislepGP) continue;
        particles.push_back(p);
      }

      
      // Retrieve clustered jets, sorted by pT, with a minimum pT cut
      //float qt = 0;
      Jets jets = apply<FastJets>(event, "jets").jetsByPt(Cuts::pT > 10*GeV && Cuts::eta < 2.5 && Cuts::eta>-1.0);  
      for (int i = 0; i < jets.size(); ++i){
	int gen_ncharged = 0;
	float gen_jet_charge = 0;
	float gen_jet_ptD = 0;
	float gen_tau10 = 0;
	float gen_tau15 = 0;
	float gen_tau20 = 0;
	float sumpt=0;
	
	FourMomentum jetmom = jets[i].momentum();
	for (const Particle& p : jets[i].particles()) {
	  //PseudoJet hfs_candidate = PseudoJet(p.px(), p.py(), p.pz(), p.energy());
	  gen_jet_ptD += pow( p.pt(), 2);
	  sumpt+=p.pt();
	  gen_tau10+=p.pt()*pow(deltaR(p,jets[i]),1);
	  gen_tau15+=p.pt()*pow(deltaR(p,jets[i]),1.5);
	  gen_tau20+=p.pt()*pow(deltaR(p,jets[i]),2);
	    
	  if (p.charge3() != 0){
	    gen_ncharged += 1;
	    gen_jet_charge += p.charge3() *p.pt(); 
	  }
	    
	}
	
	_hist_charge->fill(gen_jet_charge/jetmom.pt());
	_hist_ptD->fill(pow(gen_jet_ptD,0.5)/sumpt);
	_hist_ncharge->fill(gen_ncharged);
	_hist_tau10->fill(log(gen_tau10/jetmom.pt()));
	_hist_tau15->fill(log(gen_tau15/jetmom.pt()));
	_hist_tau20->fill(log(gen_tau20/jetmom.pt()));

	_hist_charge2D->fill(gen_jet_charge/jetmom.pt(),Q2);
	_hist_ptD2D->fill(pow(gen_jet_ptD,0.5)/sumpt,Q2);
	_hist_ncharge2D->fill(gen_ncharged,Q2);
	_hist_tau102D->fill(log(gen_tau10/jetmom.pt()),Q2);
	_hist_tau152D->fill(log(gen_tau15/jetmom.pt()),Q2);
	_hist_tau202D->fill(log(gen_tau20/jetmom.pt()),Q2);

	
      }
      
    }


    /// Normalise histograms etc., after the run
    void finalize() {
    }

    //@}


    /// The histograms.
    Histo1DPtr _hist_Q2, _hist_ncharge, _hist_charge, _hist_ptD, _hist_tau10,  _hist_tau15,  _hist_tau20;
    Histo2DPtr  _hist_ncharge2D, _hist_charge2D, _hist_ptD2D, _hist_tau102D,  _hist_tau152D,  _hist_tau202D;
    

  };


  // The hook for the plugin system
  DECLARE_RIVET_PLUGIN(DIS_JetSubs);

}
