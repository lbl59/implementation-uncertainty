# organize output objectives by mean across RDMs and across solutions

import numpy as np

obj_names = ['REL_W', 'RF_W', 'INF_NPC_W', 'PFC_W', 'WCC_W', \
        'REL_D', 'RF_D', 'INF_NPC_D', 'PFC_D', 'WCC_D', \
        'REL_F', 'RF_F', 'INF_NPC_F', 'PFC_F', 'WCC_F', \
        'REL_R', 'RF_R', 'INF_NPC_R', 'PFC_R', 'WCC_R']

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

def mean_performance_across_rdms(objs_by_rdm_dir, N_RDMS, N_SOLNS):
    """
    Calculates the mean performance of one perturbed instance across all DU SOWs.

    Parameters
    ----------
    objs_by_rdm_dir : string
        Directory where the raw DU Reevaluation output is stored.
    N_RDMS : int
        The number of DU SOWs.
    N_SOLNS : int
        The number of perturbed instancces.

    Returns
    -------
    objs_means : numpy matrix
        A matrix of dimensions N_SOLNS x 20 containing the average performance of all
        perturbed instances across all DU SOWs.

    """
    objs_matrix = np.zeros((N_SOLNS,20,N_RDMS), dtype='float')
    objs_means = np.zeros((N_SOLNS,20), dtype='float')

    for i in range(N_RDMS):
        #print('rdm: ', i)
        filepathname = objs_by_rdm_dir + str(i) + '_sols0_to_' + str(N_SOLNS) + '.csv'
        objs_file = np.loadtxt(filepathname, delimiter=",")
        objs_matrix[:,:15,i] = objs_file

        objs_file_wRegional = minimax(N_SOLNS, objs_matrix[:,:,i])

        objs_matrix[:,:,i] = objs_file_wRegional

        array_has_nan = np.isnan(np.mean(objs_matrix[:,3,i]))
        if(array_has_nan == True):
            print('NaN found at RDM ', str(i))

    for n in range(N_SOLNS):
        for n_objs in range(20):
            objs_means[n,n_objs] = np.mean(objs_matrix[n,n_objs,:])

    return objs_means

def mean_performance_across_solns(objs_by_rdm_dir, N_RDMS, N_SOLNS):
    """
    Calculates the mean performance of across all perturbed instances within one DU SOWs.

    Parameters
    ----------
    objs_by_rdm_dir : string
        Directory where the raw DU Reevaluation output is stored.
    N_RDMS : int
        The number of DU SOWs.
    N_SOLNS : int
        The number of perturbed instancces.

    Returns
    -------
    objs_means : numpy matrix
        A matrix of dimensions N_RDMS x 20 containing the average performance within
        all DU SOWs across all perturbed instances.

    """
    objs_matrix = np.zeros((N_SOLNS,20,N_RDMS), dtype='float')
    objs_means = np.zeros((N_RDMS,20), dtype='float')

    for i in range(N_RDMS):
        #print('rdm: ', i)
        filepathname = objs_by_rdm_dir + str(i) + '_sols0_to_' + str(N_SOLNS) + '.csv'
        objs_file = np.loadtxt(filepathname, delimiter=",")
        objs_matrix[:,:15,i] = objs_file
        objs_file_wRegional = minimax(N_SOLNS, objs_matrix[:,:,i])

        objs_matrix[:,:,i] = objs_file_wRegional

        array_has_nan = np.isnan(np.mean(objs_matrix[:,3,i]))
        if(array_has_nan == True):
            print('NaN found at RDM ', str(i))

    for n in range(N_RDMS):
        for n_objs in range(20):
            objs_means[n,n_objs] = np.mean(objs_matrix[:,n_objs,n])

    return objs_means

# change number of solutions available
N_SOLNS = 1000
N_RDMS = 1000

compSol = 'PW'
compSol_full = 'PW113'

# change the filepaths

objs_by_rdm_dir_p = '/home/fs02/pmr82_0001/lbl59/Implementation_Uncertainty/WaterPaths_duReeval/' + \
                    'Objectives_' + compSol + '_perturbed_May2022/Objectives_RDM'
objs_by_rdm_dir_o = '/home/fs02/pmr82_0001/lbl59/Implementation_Uncertainty/WaterPaths_duReeval/' + \
                    'Objectives_' + compSol + '_soln_Apr2022/Objectives_RDM'

fileoutpath_byRDMs = '/home/fs02/pmr82_0001/lbl59/Implementation_Uncertainty/' + \
    'post_processing_du/DU_reeval_output_Apr2022/' + compSol_full + '/meanObjs_acrossRDM_'
fileoutpath_bySoln = '/home/fs02/pmr82_0001/lbl59/Implementation_Uncertainty/' + \
    'post_processing_du/DU_reeval_output_Apr2022/' + compSol_full + '/meanObjs_acrossSoln_'
fileoutpath_byRDMs_og = '/home/fs02/pmr82_0001/lbl59/Implementation_Uncertainty/' + \
    'post_processing_du/DU_reeval_output_Apr2022/' + compSol_full + '/original_compromise_acrossRDM_'
fileoutpath_bySoln_og = '/home/fs02/pmr82_0001/lbl59/Implementation_Uncertainty/' + \
    'post_processing_du/DU_reeval_output_Apr2022/' + compSol_full + '/original_compromise_acrossSoln_'

# Comment out first two output filepaths if processing original compromise solution
outpath_byRDM = fileoutpath_byRDMs + compSol_full + '.csv'
outpath_bySoln = fileoutpath_bySoln + compSol_full + '.csv'
outpath_byRDM_og = fileoutpath_byRDMs_og + compSol_full + '.csv'
outpath_bySoln_og = fileoutpath_bySoln_og + compSol_full + '.csv'

objs_byRDM = mean_performance_across_rdms(objs_by_rdm_dir_p, N_RDMS, N_SOLNS)
objs_bySoln = mean_performance_across_solns(objs_by_rdm_dir_p, N_RDMS, N_SOLNS)
objs_ogComp_byRDM = mean_performance_across_rdms(objs_by_rdm_dir_o, 1000, 1)
objs_ogComp_bySoln = mean_performance_across_solns(objs_by_rdm_dir_o, 1000, 1)

np.savetxt(outpath_byRDM, objs_byRDM, delimiter=",")
np.savetxt(outpath_bySoln, objs_bySoln, delimiter=",")
np.savetxt(outpath_byRDM_og, objs_ogComp_byRDM, delimiter=",")
np.savetxt(outpath_bySoln_og, objs_ogComp_bySoln, delimiter=",")