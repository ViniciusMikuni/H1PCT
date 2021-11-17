#!/bin/bash
#SBATCH -C gpu
#SBATCH -q early_science
#SBATCH -n 64
#SBATCH --ntasks-per-node 4
#SBATCH --gpus-per-task 1
#SBATCH -t 300
#SBATCH -A m1759_g
#SBATCH --exclusive 

module load tensorflow

export HOROVOD_GPU_BROADCAST=MPI
export HOROVOD_GPU_ALLGATHER=MPI
export HOROVOD_GPU_ALLREDUCE=MPI
export MPICH_ALLGATHERV_PIPELINE_MSG_SIZE=0
export MPICH_MAX_THREAD_SAFETY=multiple
export MPIR_CVAR_GPU_EAGER_DEVICE_MEM=0

cd $HOME/H1/scripts
srun python Unfold_offline.py --closure --niter 50 --pct 
