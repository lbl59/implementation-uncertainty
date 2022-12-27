#!/bin/bash
#SBATCH -n 200 -N 20 -p normal
#SBATCH --job-name=rof_tables_du_reeval
#SBATCH --output=out_du/rof_tables_du_reeval.out
#SBATCH --error=out_du/rof_tables_du_reeval.err
#SBATCH --time=75:00:00
#SBATCH --mail-user=lbl59@cornell.edu
#SBATCH --mail-type=all

export OMP_NUM_THREADS=40
module load py3-mpi4py
module load py3-numpy

START="$(date +%s)"

mpirun python3 rof_tables_du_reeval.py

DURATION=$[ $(date +%s) - ${START} ]

echo ${DURATION}
