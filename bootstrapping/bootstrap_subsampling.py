# -*- coding: utf-8 -*-
"""
Created on Wed Feb 09 15:51 2022
Subsample 500 bootstrapped DU inflow, demand and evaporation SOWs from the full set of 1000 DU SOWs. 

@author: Lillian B.J. Lau
"""

import numpy as np
import pandas as pd
from ssi6 import *
import os

'''
Use the SSI6 metric to obtain the indices of 500 samples that maintain the statistical distribution 
of drought in the original set of inflows.
'''
def choose_indices(inflow_matrix, nsolns):
    chosen_idx = np.zeros(nsolns, dtype=int)

    sorted_idx = sort_by_DM(inflow_matrix)
    chosen_idx = choose_idx(sorted_idx, nsolns)
    return chosen_idx

oldpath = '/scratch/lbl59/Implementation_Uncertainty/paper3_reeval_time_series/paper3_reeval_time_series/rdm_'
newpath = '/scratch/lbl59/Implementation_Uncertainty/bootstrap_samples/rdm_'
n_rdms = 1000
nsolns = 500
sow_files = ['/inflows/', '/evaporation/','/demands/']

# Iterate over each RDM
for i in range(924, n_rdms):
    curr_rdm = str(i)

    # get indices from the Jordan Lake file from the current rdm
    jla_filepath =  oldpath + curr_rdm + '/inflows/jordan_lake_inflows.csv'
    inflows_jla = np.loadtxt(jla_filepath, delimiter=",")
    chosen_idx = choose_indices(inflows_jla, nsolns)

    # loop through the folders (inflow, evap, demands) within the rdm directory
    for j in range(len(sow_files)):
        curr_path = oldpath + curr_rdm + sow_files[j]
        out_path = newpath + curr_rdm + sow_files[j]
        if os.path.isdir(out_path) == False:
            os.makedirs(out_path)

        for filename in os.listdir(curr_path):
            f = os.path.join(curr_path, filename)

            # check if file is actually in the folder
            if os.path.isfile(f) == False:
                print("ERROR: NOT A FILE!")
                break

            original_file_df = pd.read_csv(f, sep=",", header=None, index_col=None)
            original_file_df = original_file_df.astype(float)

            original_file = original_file_df.to_numpy()
            bootstrapped_file = original_file[chosen_idx,:]

            if (bootstrapped_file.shape[0] != 500):
                print('WRONG OUTPUT FILE SIZE')

            f_out = os.path.join(out_path, filename)
            np.savetxt(f_out, bootstrapped_file, delimiter=",")
            print('RDM_', curr_rdm, ' ', sow_files[j], ' ', filename, ' done!')
