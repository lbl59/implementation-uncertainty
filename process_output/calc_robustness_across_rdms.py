# -*- coding: utf-8 -*-
"""
Created on Mon March 7 2022 16:12

@author: Lillian Bei Jia Lau

Calculates the fraction of RDMs over which each perturbed version of the solution meets all four satisficing criteria
"""
import numpy as np
import pandas as pd

obj_names = ['REL_W', 'RF_W', 'INF_NPC_W', 'PFC_W', 'WCC_W', \
        'REL_D', 'RF_D', 'INF_NPC_D', 'PFC_D', 'WCC_D', \
        'REL_F', 'RF_F', 'INF_NPC_F', 'PFC_F', 'WCC_F', \
        'REL_R', 'RF_R', 'INF_NPC_R', 'PFC_R', 'WCC_R']

utilities = ['Watertown', 'Dryville', 'Fallsland', 'Regional']

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

def satisficing(df_objs, og_soln=False):
    """
    For each DU SOW, identify if the perturbed instance of a compromise solution
    meets the three satisficing criteria.

    Parameters
    ----------
    df_objs : pandas dataframe
        Dataframe containing the average performance of each perturbed instance across
        all DU SOWs.
    og_soln : boolean, optional
        Set to true if this is the performance of the original compromise solution
        across all DU SOWs. The default is False.

    Returns
    -------
    df_objs : pandas dataframe
        The input dataframe, but with four new columns containing 1's or 0's depending on 
        whether the perturbed instance meets the satisficing criteria.

    """
    for i in range(4):
        if og_soln == False:
            df_objs['satisficing_W'] = (df_objs['REL_W'] >= 0.98).astype(int) *\
                                        (df_objs['WCC_W'] <= 0.10).astype(int) *\
                                        (df_objs['RF_W'] <= 0.10).astype(int)

            df_objs['satisficing_D'] = (df_objs['REL_D'] >= 0.98).astype(int) *\
                                        (df_objs['WCC_D'] <= 0.10).astype(int) *\
                                        (df_objs['RF_D'] <= 0.10).astype(int)

            df_objs['satisficing_F'] = (df_objs['REL_F'] >= 0.98).astype(int) *\
                                        (df_objs['WCC_F'] <= 0.10).astype(int) *\
                                        (df_objs['RF_F'] <= 0.10).astype(int)
            '''
            df_objs['satisficing_R'] = (df_objs['REL_R'] >= 0.98).astype(int) *\
                                        (df_objs['WCC_R'] < 0.10).astype(int) *\
                                        (df_objs['RF_R'] < 0.10).astype(int)
                                        #(df_objs['inf_diff_R'] < 10)
            '''
        else:
            df_objs['satisficing_W'] = (df_objs['REL_W'] >= 0.98).astype(int)*\
                                        (df_objs['WCC_W'] <= 0.10).astype(int)*\
                                        (df_objs['RF_W'] <= 0.10).astype(int)

            df_objs['satisficing_D'] = (df_objs['REL_D'] >= 0.98).astype(int)*\
                                        (df_objs['WCC_D'] <= 0.10).astype(int)*\
                                        (df_objs['RF_D'] <= 0.10).astype(int)

            df_objs['satisficing_F'] = (df_objs['REL_F'] >= 0.98).astype(int)*\
                                        (df_objs['WCC_F'] <= 0.10).astype(int)*\
                                        (df_objs['RF_F'] <= 0.10).astype(int)

    df_objs['satisficing_R'] = df_objs[['satisficing_W', 'satisficing_D', 'satisficing_F']].min()
    
    return df_objs

def calc_robustness(objs_by_rdm_dir_pt, objs_by_rdm_dir_og, N_RDMS, N_SOLNS):
    """
    Calculates the robustness of each perturbed instance across all DU SOWs.

    Parameters
    ----------
    objs_by_rdm_dir_pt : string
        Directory to the location where the performance objective values of the 
        perturbed instances are stored.
    objs_by_rdm_dir_og : string
        Directory to the location where the performance objective values of the 
        original compromise solution are stored.
    N_RDMS : int
        Number of DU SOWs.
    N_SOLNS : int
        Number of perturbed instances.

    Returns
    -------
    solution_robustness : numpy matrix
        An array of size (N_SOLNS+1) x 4 of the robustness of each utility across all 
        DU SOWs for each perturbed instance and the original compromise solution.

    """
    # matrix structure: (N_SOLNS, N_OBJS, N_RDMS)
    objs_matrix_pt = np.zeros((N_SOLNS,20,N_RDMS), dtype='float')
    objs_matrix_og = np.zeros((N_RDMS,20), dtype='float')

    satisficing_matrix = np.zeros((N_SOLNS+1,4,N_RDMS), dtype='float')
    solution_robustness = np.zeros((N_SOLNS+1,4), dtype='float')

    for i in range(N_RDMS):
        # get one perturbed instance's behavior over all RDMs
        filepathname_pt = objs_by_rdm_dir_pt + str(i) + '_sols0_to_' + str(N_SOLNS) + '.csv'
        filepathname_og = objs_by_rdm_dir_og + str(i) + '_sols0_to_1.csv'

        objs_file_pt = np.loadtxt(filepathname_pt, delimiter=",")
        objs_file_og = np.loadtxt(filepathname_og, delimiter=",")

        objs_matrix_pt[:,:15,i] = objs_file_pt
        objs_matrix_og[i,:15] = objs_file_og

        objs_file_wRegional_pt = minimax(N_SOLNS, objs_matrix_pt[:,:,i])
        objs_file_wRegional_og = minimax(1, objs_matrix_og[:])

        objs_matrix_pt[:,:,i] = objs_file_wRegional_pt
        objs_matrix_og[:,:] = objs_file_wRegional_og
        '''
        # NaN check
        array_has_nan = np.isnan(np.mean(objs_matrix_pt[:,3,i]))
        if(array_has_nan == True):
            print('NaN found at RDM ', str(i))
        '''
    # for the perturbed instances
    for r in range(N_RDMS):

        df_curr_rdm_pt = pd.DataFrame(objs_matrix_pt[:, :, r], columns = obj_names)
        df_curr_rdm_og = pd.DataFrame(objs_matrix_og, columns = obj_names)

        df_curr_rdm = satisficing(df_curr_rdm_pt, og_soln=False)

        satisficing_matrix[:1000,:,r] = df_curr_rdm.iloc[:,20:24]

    # for the original compromise
    df_curr_rdm_og = pd.DataFrame(objs_matrix_og, columns = obj_names)
    df_curr_rdm_og = satisficing(df_curr_rdm_og, og_soln=True)

    satisficing_matrix[1000,:,:] = (df_curr_rdm_og.iloc[:,20:24].values).T

    for n in range(N_SOLNS+1):
        solution_robustness[n,0] = np.sum(satisficing_matrix[n,0,:])/N_RDMS
        solution_robustness[n,1] = np.sum(satisficing_matrix[n,1,:])/N_RDMS
        solution_robustness[n,2] = np.sum(satisficing_matrix[n,2,:])/N_RDMS

    solution_robustness[:,3] = np.min(solution_robustness[:,:3], axis=1)

    return solution_robustness

# change these ones, depending on solution chosen
N_RDMS = 1000
N_SOLNS = 1000

compSol = 'PW'
compSol_full = 'PW113'

objs_by_rdm_dir_p = '/home/fs02/pmr82_0001/lbl59/Implementation_Uncertainty/' +\
    'WaterPaths_duReeval/Objectives_' + compSol + '_perturbed_May2022/Objectives_RDM'
objs_by_rdm_dir_o = '/home/fs02/pmr82_0001/lbl59/Implementation_Uncertainty/' +\
    'WaterPaths_duReeval/Objectives_' + compSol + '_soln_Apr2022/Objectives_RDM'

fileoutpath_robustness = '/home/fs02/pmr82_0001/lbl59/Implementation_Uncertainty/' +\
    'post_processing_du/DU_reeval_output_Apr2022/' + compSol_full + '/robustness_'

# change this one, depending on type of solution being processeed
# last line contains the robustness of the original solution
filename = 'perturbed_og_' + compSol_full + '.csv'

outpath = fileoutpath_robustness + filename

robustness_perturbation_og = calc_robustness(objs_by_rdm_dir_p, objs_by_rdm_dir_o, \
                                             N_RDMS, N_SOLNS)

np.savetxt(outpath, robustness_perturbation_og, delimiter=",")
