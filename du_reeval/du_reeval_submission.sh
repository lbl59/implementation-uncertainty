#!/bin/bash
#SBATCH -N 10 -n 200 -p normal
#SBATCH --tasks-per-node 20
#SBATCH --job-name=du_reeval_LS98
#SBATCH --output=out_du/du_reeval_LS98.out
#SBATCH --error=out_du/du_reeval_LS98.err
#SBATCH --time=200:00:00
#SBATCH --mail-user=lbl59@cornell.edu
#SBATCH --mail-type=ALL

export OMP_NUM_THREADS=40
module load py3-mpi4py
module load py3-numpy

START="$(date +%s)"

mpirun python3 du_reeval_script.py

DURATION=$[ $(date +%s) - ${START} ]

echo ${DURATION}
