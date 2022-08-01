# -*- coding: utf-8 -*-
"""
Finds the solution with the lowest robustness for each utility
Created on Tue Apr  5 17:36:28 2022

@author: lbl59
"""
import numpy as np
import pandas as pd

obj_names = ['REL_W', 'RF_W', 'INF_NPC_W', 'PFC_W', 'WCC_W', \
        'REL_D', 'RF_D', 'INF_NPC_D', 'PFC_D', 'WCC_D', \
        'REL_F', 'RF_F', 'INF_NPC_F', 'PFC_F', 'WCC_F', \
        'REL_R', 'RF_R', 'INF_NPC_R', 'PFC_R', 'WCC_R']

N_SOLNS = 1000
N_RDMS = 1000

def find_min_robustness_idx(robustness_arr):
    """
    Finds the index of the perturbed instance with the lowest robustness for each 
    utility and the region.

    Parameters
    ----------
    robustness_arr : numpy array
        Array of floats containing the values of the robustness across all SOWs 
        for all utilities and the region.

    Returns
    -------
    watertown : int
        Index of the perturbed instance resulting in the worst robustness for Watertown.
    dryville : int
        Index of the perturbed instance resulting in the worst robustness for Dryville.
    fallsland : int
        Index of the perturbed instance resulting in the worst robustness for Fallsland.
    regional : int
        Index of the perturbed instance resulting in the worst robustness for the region.

    """

    # solution with min robustness for Watertown
    watertown = np.argmin(robustness_arr[:1000,0])

    # solution with min robustness for Dryville
    dryville = np.argmin(robustness_arr[:1000,1])

    # solution with min robustness for Fallsland
    fallsland = np.argmin(robustness_arr[:1000,2])

    # solution with min robustness for the region
    regional = np.argmin(robustness_arr[:1000,3])

    return watertown, dryville, fallsland, regional

def minimax(N_SOLNS, objs):
    """
    Performs regional minimax.

    Parameters
    ----------
    N_SOLNS : int
        Number of perturbed instances.
    objs : numpy matrix
        Performance objectives matrix WITHOUT regional performance values.

    Returns
    -------
    objs : numpy matrix
        Performance objectives matrix WITH regional performance values.

    """
    for i in range(N_SOLNS):
        for j in range(5):
            if j == 0:
                objs[i,15] = np.min([objs[i,0],objs[i,5], objs[i,10]])
            else:
                objs[i, (j+15)] = np.max([objs[i,j],objs[i,j+5], objs[i,j+10]])
    return objs

def gather_objs_across_rdms(idx, objs_by_rdm_dir, N_RDMS):
    """
    Gathers the values of performance objectives for one perturbed instance 
    across the 1000 DU SOWs.

    Parameters
    ----------
    idx : int
        The row-index of the perturbed instance.
    objs_by_rdm_dir : string
        The directory where the output file will be stored.
    N_RDMS : int
        The total number of DU SOWs.

    Returns
    -------
    df_idx_acrossRDM : numpy array
        The performance of one perturbed instance across 1000 DU SOWs.

    """
    objs_idx = np.zeros((N_RDMS,20), dtype='float')
    objs_file_wRegional = objs_idx
    for i in range(N_RDMS):
        filepathname = objs_by_rdm_dir + str(i) + '_sols0_to_' + str(N_SOLNS) + '.csv'

        objs_file = np.loadtxt(filepathname, delimiter=",")

        objs_idx[i,:15] = objs_file[idx,:]
        if i == 0:
            print(idx)
            print(objs_file[idx,:])
    objs_file_wRegional = minimax(N_RDMS, objs_idx)

    objs_idx = objs_file_wRegional

    df_idx_acrossRDM = pd.DataFrame(objs_idx, columns = obj_names)

    return df_idx_acrossRDM


'''
Start processing output here. Get the following:
1. Indices of each utility’s and the region’s worst robustness
2. DV values of the solution with the worst regional robustness
3. Avg objective values of the solution with the worst regional robustness
4. How the performance of this solution varies across different SOWs
'''
# change these filenames depending on which solution is being examined
compSol = 'PW'
compSol_full = 'PW113'

main_filepath = '/home/fs02/pmr82_0001/lbl59/Implementation_Uncertainty/'+\
    'post_processing_du/DU_reeval_output_Apr2022/' + compSol_full + '/'

robustness_inputfile = main_filepath + 'robustness_perturbed_og_' + compSol_full + '.csv'

perturbed_soln_file = '/home/fs02/pmr82_0001/lbl59/Implementation_Uncertainty/'+\
    'WaterPaths_duReeval/IU_Samples/IU_allMeasures_' + compSol + '.csv'

perturbed_objs_file = main_filepath + 'meanObjs_acrossRDM_' + compSol_full + '.csv'

# change these filepaths depending on which compromise is being examined
objs_by_rdm_dir_p = '/home/fs02/pmr82_0001/lbl59/Implementation_Uncertainty/' +\
    'WaterPaths_duReeval/Objectives_' + compSol + '_perturbed_Apr2022/Objectives_RDM'
objs_by_rdm_dir_o = '/home/fs02/pmr82_0001/lbl59/Implementation_Uncertainty/' +\
    'WaterPaths_duReeval/Objectives_' + compSol + '_soln_Apr2022/Objectives_RDM'

robustness_arr = np.loadtxt(robustness_inputfile, delimiter=",")
perturbed_soln_arr = np.loadtxt(perturbed_soln_file, delimiter=",")
perturbed_objs_arr = np.loadtxt(perturbed_objs_file, delimiter=",")

min_idx_w, min_idx_d, min_idx_f, min_idx_r = find_min_robustness_idx(robustness_arr)

min_robustness_idx = {'Utility':['Watertown', 'Dryviile', 'Fallsland', 'Regional'], \
                  'Sol_num': [min_idx_w, min_idx_d, min_idx_f, min_idx_r]}
                  
'''
1. Indices of each utility's and the region's worst robustness
'''
robustness_outputfile_min = main_filepath + 'worst_robustness_idx_' + compSol_full + '.csv'
min_robustness_idx = pd.DataFrame(min_robustness_idx, index=None)
min_robustness_idx.to_csv(robustness_outputfile_min, sep=",", index=None)

'''
2. DV values of the solution with the worst robustness
'''

# Watertown
min_dvs_idx_w = perturbed_soln_arr[min_idx_w,:]
min_dvs_filepath_w = main_filepath + 'worst_robustness_regDVs_W_' + \
    compSol_full + '.csv'
np.savetxt(min_dvs_filepath_w, min_dvs_idx_w)

# Dryville
min_dvs_idx_d = perturbed_soln_arr[min_idx_d,:]
min_dvs_filepath_d = main_filepath + 'worst_robustness_regDVs_D_' + \
    compSol_full + '.csv'
np.savetxt(min_dvs_filepath_d, min_dvs_idx_d)

# Fallsland
min_dvs_idx_f = perturbed_soln_arr[min_idx_f,:]
min_dvs_filepath_f = main_filepath + 'worst_robustness_regDVs_F_' + \
    compSol_full + '.csv'
np.savetxt(min_dvs_filepath_f, min_dvs_idx_f)

# Regional
min_dvs_idx_r = perturbed_soln_arr[min_idx_r,:]
min_dvs_filepath_r = main_filepath + 'worst_robustness_regDVs_R_' + \
    compSol_full + '.csv'
np.savetxt(min_dvs_filepath_r, min_dvs_idx_r)

'''
3. Avg objective values of the solution with the worst robustness
'''
# Watertown
min_obj_idx_w = perturbed_objs_arr[min_idx_w,:]
min_obj_filepath_w = main_filepath + 'worst_robustness_avgObj_W_' + compSol_full + '.csv'
np.savetxt(min_obj_filepath_w, min_obj_idx_w)

# Dryville
min_obj_idx_d = perturbed_objs_arr[min_idx_d,:]
min_obj_filepath_d = main_filepath + 'worst_robustness_avgObj_D_' + compSol_full + '.csv'
np.savetxt(min_obj_filepath_d, min_obj_idx_d)

# Fallsland
min_obj_idx_f = perturbed_objs_arr[min_idx_f,:]
min_obj_filepath_f = main_filepath + 'worst_robustness_avgObj_F_' + compSol_full + '.csv'
np.savetxt(min_obj_filepath_f, min_obj_idx_f)

# Regional
min_obj_idx_r = perturbed_objs_arr[min_idx_r,:]
min_obj_filepath_r = main_filepath + 'worst_robustness_avgObj_R_' + compSol_full + '.csv'
np.savetxt(min_obj_filepath_r, min_obj_idx_r)

'''
4. How the performance of this solution varies across different SOWs
'''

# Watertown
worst_robustness_objs_acrossRDM_w = gather_objs_across_rdms(min_idx_w, objs_by_rdm_dir_p, \
                                                          objs_by_rdm_dir_o, N_RDMS)
min_perturbed_objs_acrossRDMs_filepath_w = main_filepath + 'worst_robustness_objs_acrossRDMs_W_' + \
    compSol_full + '.csv'
worst_robustness_objs_acrossRDM_w.to_csv(min_perturbed_objs_acrossRDMs_filepath_w, sep=",", index=None)

# Dryville
worst_robustness_objs_acrossRDM_d = gather_objs_across_rdms(min_idx_d, objs_by_rdm_dir_p, \
                                                          objs_by_rdm_dir_o, N_RDMS)
min_perturbed_objs_acrossRDMs_filepath_d = main_filepath + 'worst_robustness_objs_acrossRDMs_D_' + \
    compSol_full + '.csv'
worst_robustness_objs_acrossRDM_d.to_csv(min_perturbed_objs_acrossRDMs_filepath_d, sep=",", index=None)

# Fallsland
worst_robustness_objs_acrossRDM_f = gather_objs_across_rdms(min_idx_f, objs_by_rdm_dir_p, \
                                                          objs_by_rdm_dir_o, N_RDMS)
min_perturbed_objs_acrossRDMs_filepath_f = main_filepath + 'worst_robustness_objs_acrossRDMs_F_' + \
    compSol_full + '.csv'
worst_robustness_objs_acrossRDM_f.to_csv(min_perturbed_objs_acrossRDMs_filepath_f, sep=",", index=None)

# Regional
worst_robustness_objs_acrossRDM_r = gather_objs_across_rdms(min_idx_r, objs_by_rdm_dir_p, \
                                                          objs_by_rdm_dir_o, N_RDMS)
min_perturbed_objs_acrossRDMs_filepath_r = main_filepath + 'worst_robustness_objs_acrossRDMs_R_' + \
    compSol_full + '.csv'
worst_robustness_objs_acrossRDM_r.to_csv(min_perturbed_objs_acrossRDMs_filepath_r, sep=",", index=None)

