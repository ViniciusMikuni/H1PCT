# H1 Jet substructure with Omnifold

Repository to store the scripts used in the studies of jet observable unfolding using the H1 dataset. Main branch stores the strategy planned for the current iteration, while the ```perlmutter``` branch is used for development of Omnifold at scale. 

Current datasets are stored both at NERSC and at ML4HEP local machines. Those are already curated ```.h5``` files containing per particle and per jet information. To run Omnifold use:

```bash
python Unfold_offline.py  --data_folder FOLDER/CONTAINING/H5/FILES [--closure] [--pct] --niter NITER
```

The flag ```closure``` runs the unfolding procedure without data files, but taking the Djangoh simulation as the data representative. To run with the PCT model instead of MLPs use the flag ```pct```.

The outputs are the trained models for steps 1 and 2 for each omnifold iteration up to NITER. Plotting scripts are provided to compare unfolded data and simulation (```Plot_results.pt```), reconstructed data vs MC (```Plot_reco.py```), and Q2 dependence for the inputs using data ```Plot_q2.py```. 