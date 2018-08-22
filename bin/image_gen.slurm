#!/bin/bash
#SBATCH -t 1:00:00
#SBATCH --mem=16G
#SBATCH -N 8
#SBATCH -n 96
#SBATCH -c 1
#SBATCH -p compute

source /work/BuschU/mlxd_2018/4vtx_chaos/conda/mc2/etc/profile.d/conda.sh
conda activate 4vtx_chaos
module load openmpi.gcc/1.10.1

mpirun -n 96 python /work/BuschU/mlxd_2018/4vtx_chaos/gpue/py/mpi_vis_dual.py $1 $2 true true
