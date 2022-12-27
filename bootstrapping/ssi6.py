# -*- coding: utf-8 -*-
"""
Created on Wed Feb 09 15:51 2022

@author: Lillian B.J. Lau
"""

import numpy as np
import pandas as pd

def ssi6(inflow_arr):
    timesteps = len(inflow_arr)
    inflow_arr = np.where(inflow_arr < 10**(-10), 10**(-10), inflow_arr)
    # log-normalize the inflow file
    ssi6 = np.zeros((1, timesteps), dtype=float)
    # standardization / whitening of each of the 1000 realizations
    #realization_k = np.where(inflow_file[k, :] == 0, 0.000000000000001, inflow_file[k, :])
    inflow_pd = pd.DataFrame(np.log10(inflow_arr).flatten()).fillna(method='pad')

    mu = inflow_pd.mean()
    sigma = inflow_pd.std()
    Z_k = (inflow_pd - mu) / sigma

    # SSI_6 calculations
    rolling_avg = (Z_k.rolling(24, min_periods=1).mean()).fillna(method='pad')
    ssi6_arr = (rolling_avg.to_numpy()).flatten()
    #print("SSI6 realization ", k, " done")

    return ssi6_arr

# function to check if the window meets the conditions for a drought
# drought is defined as a continuous period of at least 3 months where SSI_6 < 0 and hits -1
def meets_conditions(window):
    if np.all(window < 0) and (np.min(window) <= -1):
        return 1
    else:
        return 0

def find_max_DM(ssi6_arr):
    drought_severity = []
    for i in range(len(ssi6_arr) - 12):
        window = ssi6_arr[i: i+12]
        if meets_conditions(window) == 1:
            DM = (np.sum(window)) * (-1)
            drought_severity.append(DM)
    return np.max(drought_severity)

def sort_by_DM(inflow_file):
    realizations = inflow_file.shape[0]
    timesteps = inflow_file.shape[1]
    max_DMs = np.zeros((realizations,2), dtype=float)
    max_DMs_df = pd.DataFrame(max_DMs, columns = ['DM_val', 'DM_idx'], index=None)

    max_DMs_df['DM_idx'] = np.arange(0,realizations)
    # settle DM and ssi6 by realization
    for i in range(realizations):
        inflow_arr = inflow_file[i,:]
        ssi6_arr = ssi6(inflow_arr)
        max_DMs_df.loc[i,'DM_val'] = find_max_DM(ssi6_arr)

    sorted_DMs = max_DMs_df.sort_values(by=['DM_val'])
    sorted_idx = sorted_DMs['DM_idx'].values
    #print('sorted_idx =', sorted_idx)

    return sorted_idx

def choose_idx(sorted_idx, nsolns):
    # for a given input file, select the first and last index
    # then sample from the remaining indices
    realizations = len(sorted_idx)
    chosen_idx = np.zeros(nsolns, dtype=int)

    step_size = int(realizations/nsolns)
    '''
     # sanity check
    if (step_size == 2):
        print("CORRECT; PLEASE PROCEED")
    else:
        print("WRONG STEP SIZE; STOP RIGHT THERE!!")
    '''

    chosen_idx[0]= sorted_idx[0]
    chosen_idx[1]= sorted_idx[realizations-1]
    for i in range(2, nsolns):
        chosen_idx[i] = sorted_idx[(i*2) - 1]
    return chosen_idx

'''
inflow_matrix = np.loadtxt("cane_creek_inflows.csv", delimiter=",")

nsolns = 500
realizations = inflow_matrix.shape[0]
new_inflow_matrix = np.zeros((realizations, nsolns), dtype=float)

sorted_idx = sort_by_DM(inflow_matrix)
print(sorted_idx)
chosen_idx = choose_idx(sorted_idx, nsolns)
new_inflow_matrix = inflow_matrix[chosen_idx,:]

np.savetxt('cane_creek_inflows500.csv', new_inflow_matrix, delimiter=",")
'''
