# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 2022 10:36

@author: Lillian Bei Jia Lau
"""

from mpi4py import MPI
import numpy as np
import subprocess, sys, time
import os

# 20 nodes, 50 RDMs per node
# 10 tasks per node
# 5 RDMs per task
comm = MPI.COMM_WORLD
rank = comm.Get_rank() # up to 200 processes
print('rank = ', rank)

N_RDMs = 10  # only 10 RDMs used due to memory restrictions
N_REALIZATIONS = 500   # number of bootstrapped realizations
N_TASKS_PER_NODE = 10 # rank ranges from 0 to 200
N_RDMS_PER_TASK = 5 # each task handles five RDMs
N_TASKS = 200 # 200

DATA_DIR = "/home/fs02/pmr82_0001/lbl59/Implementation_Uncertainty/"
SOLS_FILE_NAME = "IU_Samples/LS_soln.csv"   # LS for least-squares, PW for power index
N_SOLS = 1

OMP_NUM_THREADS = 40

for i in range(N_RDMS_PER_TASK):
    current_RDM = rank + (N_TASKS * i)

    command_gen_tables = "./waterpaths -T {} -t 2344 -r {} -d {} -C 1 -O rof_tables_duReeval/rdm_{} -e 0 \
            -U TestFiles/rdm_utilities_test_problem_reeval.csv \
            -W TestFiles/rdm_water_sources_test_problem_reeval.csv \
            -P TestFiles/rdm_dmp_test_problem_reeval.csv \
            -s {} -f 0 -l {} -R {}\
            -p false".format(OMP_NUM_THREADS, N_REALIZATIONS, DATA_DIR, current_RDM, SOLS_FILE_NAME, N_SOLS, current_RDM)

    print(command_gen_tables)
    os.system(command_gen_tables)


comm.Barrier()
