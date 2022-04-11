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

     // book(_hist_Q2, "Q2",logspace(10,0.1, 1000.0));
     // book(_hist_y, "y",10,0.,1.);
     // book(_hist_x, "xBj",logspace(10,0.00001, 1.0));
     // book(_hist_ept, "ept", logspace(6,10,100));

      //book(_h["jetpt"], 1, 1, 1);
      //book(_h["jeteta"], 2, 1, 1);
      //book(_h["qt"], 3, 1, 1);
      //book(_h["dphi"], 4, 1, 1);
      
      Histo1DPtr tmp;

/*      
      _h_Q2_jetpt.add( 150.,  237.,  book(tmp, "jetpt_150", {10, 16.050000953674196, 24.07000095367545, 32.80000095367653, 43.14000095367447, 57.160000953671684, 105.12264}));
      _h_Q2_jetpt.add( 237.,346.5 ,  book(tmp, "jetpt_237", {10, 16.650001907348607, 26.420001907350134, 37.17000190734998, 51.74000190734708, 78.69000190735215, 83.03000190735438, 100.90953}));
      _h_Q2_jetpt.add( 346.5,51932.,book(tmp, "jetpt_346", {10, 21.190002861023633, 36.510002861024425, 58.74000286102, 71.57000286102283, 84.46000286102942, 99.16000286103694, 117.95000286104656, 143.86293}));
      
      _h_Q2_jeteta.add( 150.,  237.,  book(tmp, "jeteta_150", {-1, -0.5218995231628945, -0.2855995231629205, -0.042099523162935776, 0.2636004768370517, 0.7280004768370005, 2.4999971}));
      _h_Q2_jeteta.add( 237.,346.5 ,  book(tmp, "jeteta_237", {-1, -0.4987999403954104, -0.2892999403954335, -0.10419994039545098, 0.07640005960455212, 0.2666000596045379, 0.46590005960451597, 0.7634000596044832, 2.4999995}));
      _h_Q2_jeteta.add( 346.5,51932.,book(tmp, "jeteta_346", {-1, -0.11549982118616176, 0.1602001788138392, 0.40660017881381205, 0.6466001788137856, 1.0048001788137462, 2.4999936}));

      _h_Q2_dphi.add( 150.,  237.,  book(tmp, "dphi_150", {0, 0.14950005215406403, 0.558100052154019, 0.8652000521539852, 1.1659000521539522, 1.5232000521539129, 1.570620059967041}));
      _h_Q2_dphi.add( 237.,346.5 ,  book(tmp, "dphi_237", {0, 0.09770005960464655, 0.3703000596046203, 0.5800000596045972, 0.8009000596045729, 1.01040005960455, 1.2351000596045252, 1.473000059604499, 1.5705121755599976}));
      _h_Q2_dphi.add( 346.5,51932.,book(tmp, "dphi_346", {0.0, 0.06160000000000074, 0.3050999999999827, 0.5131999999999598, 0.728899999999936, 0.9413999999999126, 1.156699999999889, 1.3817999999998642, 1.5706232786178589}));

      _h_Q2_qt.add( 150.,  237.,  book(tmp, "qt_150", {0, 0.2626663129903716, 0.7018663129903232, 1.249766312990263, 1.8557663129901962, 2.745866312991754, 4.381366312993512, 8.171009}));
      _h_Q2_qt.add( 237.,346.5 ,  book(tmp, "qt_237", {0, 0.2774143999677533, 0.6396143999677134, 1.071414399967666, 1.6017143999676076, 2.2541143999681, 3.6533143999710527, 5.329314399968686, 6.249228})); 
      _h_Q2_qt.add( 346.5,51932.,book(tmp, "qt_346", {0, 0.3009084919101769, 0.7532084919101271, 1.2760084919100696, 2.0770084919101524, 5.054179}));
      
      _h_y_jetpt.add( 0.2,  0.31,  book(tmp, "jetpt_y_02", {10.000006, 18.890005722046222, 32.90000572204809, 71.07000572204552, 84.75000572205252, 112.21000572206657, 124.81343}));
      _h_y_jetpt.add( 0.31, 0.44,  book(tmp, "jetpt_y_03", {10.000003, 24.090002861024086, 75.56000286102487, 91.68000286103312, 109.07000286104201, 139.06075}));
      _h_y_jetpt.add( 0.44, 0.7 ,  book(tmp, "jetpt_y_04", {10.000004, 26.590003814698793, 46.400003814696774, 57.88000381469449, 69.48000381469608, 81.47000381470221, 92.3900038147078, 104.41000381471395, 118.09000381472094, 131.91000381472244, 143.03601}));
      
      _h_y_jeteta.add( 0.2,  0.31,  book(tmp, "jeteta_y_02", {-0.5050307, -0.07563069162372818, 0.23276930837626364, 0.5192693083762321, 0.8820693083761921, 1.5605693083761174, 2.499989}));
      _h_y_jeteta.add( 0.31, 0.44,  book(tmp, "jeteta_y_03", {-0.8509098, -0.42630982913975624, -0.1543098291397862, 0.1611901708602117, 0.6254901708601606, 2.4999971}));
      _h_y_jeteta.add( 0.44, 0.7 ,  book(tmp, "jeteta_y_04", {-0.9999995, -0.6128995231628844, -0.3632995231629119, -0.09299952316293723, 0.5558004768370195, 2.4999936}));
      
      _h_y_dphi.add( 0.2,  0.31,  book(tmp, "dphi_y_02", {2.9802322e-08, 0.1026000298023243, 0.5320000298022801, 0.8518000298022449, 1.152100029802212, 1.4954000298021741, 1.5705938339233398}));
      _h_y_dphi.add( 0.31, 0.44,  book(tmp, "dphi_y_03", {0.0, 0.2017999999999941, 0.6337999999999465, 0.9443999999999123, 1.2432999999998795, 1.5619999999998444, 1.570620059967041}));
      _h_y_dphi.add( 0.44, 0.7 ,  book(tmp, "dphi_y_04", {5.9604645e-08, 0.44800005960461176, 0.8419000596045684, 1.1664000596045327, 1.5161000596044942, 1.570563793182373}));
      
      _h_y_qt.add( 0.2,  0.31,  book(tmp, "qt_y_02", {4.7180823e-05, 0.23574718082344825, 0.5346471808234153, 0.8288471808233829, 1.183647180823344, 1.6080471808232972, 2.290547180823867, 3.4194471808262494, 4.967163}));
      _h_y_qt.add( 0.31, 0.44,  book(tmp, "qt_y_03", {4.4488774e-05, 0.3935444887736833, 0.9438444887736227, 1.5278444887735585, 2.363744488774274, 3.621344488776928, 5.4841785}));
      _h_y_qt.add( 0.44, 0.7 ,  book(tmp, "qt_y_04", {1.732245e-05, 0.2536173224507191, 0.6306173224506776, 1.0896173224506271, 1.645517322450566, 2.33801732245124, 3.161017322452977, 4.18321732245432, 6.2783265}));
*/

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
      double x  = dk.x();
      double y  = dk.y();
      double Q2 = dk.Q2();

      if (Q2 < 150.0*GeV2) vetoEvent;
      if ( y>0.7) vetoEvent;
      if(y<0.2) vetoEvent;
      
      // Weight of the event
      //_hist_Q2->fill(Q2);
      //_hist_y->fill(y);
      //_hist_x->fill(x);

      // Momentum of the scattered lepton
      const DISLepton& dl = apply<DISLepton>(event,"Lepton");
      if ( dl.failed() ) return;
      const FourMomentum leptonMom = dl.out();
      const double ptel = leptonMom.pT();
      const double enel = leptonMom.E();

      if(enel<11*GeV) vetoEvent;
      
      //const double thel = leptonMom.angle(dk.beamHadron().mom())/degree;
      //_hist_ept->fill(ptel);

/*
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
*/
      
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


  RIVET_DECLARE_PLUGIN(H1_2022_TMD);

}
