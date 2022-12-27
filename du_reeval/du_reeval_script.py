# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 10:49:25 2021

@author: lbl59
"""

from mpi4py import MPI
import numpy as np
import subprocess, sys, time
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

N_RDMs = 10  # only 10 RDMs used due to memory restrictions

OMP_NUM_THREADS = 40
N_REALIZATIONS = 500
DATA_DIR = "/home/fs02/pmr82_0001/lbl59/Implementation_Uncertainty/WaterPaths_duReeval/"

SOLS_FILE_NAME = "IU_Samples/IU_allMeasures_LS.csv"   # LS for least-squares, PW for power index

N_NODES = 10
N_TASKS_PER_NODE = 20
N_TASKS = int(N_TASKS_PER_NODE * N_NODES) # should be 200
N_RDMS_PER_TASK = int(N_RDMs/N_TASKS)  # should be 50

SOL_NUM = 500   # change this number

for i in range(N_RDMS_PER_TASK):
    current_RDM = rank + (N_TASKS * i)

    command_run_rdm = "./waterpaths -T {} -t 2344 -r {} -d {} -C -1 -O rof_tables_duReeval/rdm_{} -e 0 \
        -U TestFiles/rdm_utilities_test_problem_reeval.csv \
        -W TestFiles/rdm_water_sources_test_problem_reeval.csv \
        -P TestFiles/rdm_dmp_test_problem_reeval.csv \
        -R {} -s {} -f 0 -l {} -p false -c false".format(OMP_NUM_THREADS, N_REALIZATIONS, DATA_DIR, current_RDM, current_RDM, SOLS_FILE_NAME, SOL_NUM)

    print(command_run_rdm)
    os.system(command_run_rdm)

comm.Barrier()
