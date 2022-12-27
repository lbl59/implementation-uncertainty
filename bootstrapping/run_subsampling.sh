#!/bin/bash
#SBATCH -n 32 -N 20 -p normal
#SBATCH --job-name=subsample_files
#SBATCH --time=24:00:00
#SBATCH --mail-user=lbl59@cornell.edu
#SBATCH --mail-type=all
#SBATCH --output=output/subsamples500.out
#SBATCH --error=output/subsamples500.err
#SBATCH --mail-user=lbl59@cornell.edu
#SBATCH --mail-type=all

START="$(date +%s)"

python bootstrap_subsampling.py

DURATION=$[ $(date +%s) - ${START} ]

echo ${DURATION}