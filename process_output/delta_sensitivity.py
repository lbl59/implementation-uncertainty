# -*- coding: utf-8 -*-
"""
Created on Mon Apr  18 17:36:28 2022
Conducts Delta moment-independent sensitivity analysis for both the performance objectives 
and robustness against changes in decision variables and DU factor multipliers
@author: lbl59
"""

from SALib.analyze import delta
import numpy as np
import pandas as pd

def find_bounds(input_file):
    """
    Finds the founds of the decision variables or DU factor multipliers.

    Parameters
    ----------
    input_file : numpy matrix
        A numpy matrix that specifies the lower and upper bounds of each decision
        variable or DU factor multiplier.

    Returns
    -------
    bounds : tuple
        The lower and upper bound of a decision variable or DU factor multiplier.

    """
    bounds = np.zeros((input_file.shape[1],2), dtype=float)
    for i in range(input_file.shape[1]):
        bounds[i,0] = min(input_file[:,i])
        bounds[i,1] = max(input_file[:,i])

    return bounds

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

def delta_sensitivity(dv_du, measured_outcomes, names, mo_names, bounds, compSol_full, rob_objs):
    """
    Main function that performs Delta moment-independent sensitivity analysis
    Writes a csv file to a subfolder named 'delta_[rob_objs]_[mode]/S1_[util]_[compSol].csv'
    
    Parameters
    ----------
    dv_du : numpy matrix
        Contains the float values of the decision variables of each perturbed instance
        of a given compromise solution OR the DU factor multipliers.
    measured_outcomes : numpy matrix
        Contains the float values of either the performance objectives or robustness 
        values, depending on the mode.
    names : list of strings
        Names of all relevant decision variables or DU factor multipliers.
    mo_names : list of strings
        Names of all relevant performance objectives or utilities (for robustness).
    bounds : numpy matrix
        An (len(dv_du) x 2) matrix of the lower and upper bounds of the decision variables
        or DU factor multipliers.
    compSol_full : string
        Longer abbreviation of the compromise solution name
        Social planner: LS98
        Pragmatist: PW113
    rob_objs : string
        Subfolder label indicating if this is sensitivity of robustness or objectives.

    Returns
    -------
    None.

    """
    X = dv_du
    Y = measured_outcomes

    problem = {
        'num_vars': int(dv_du.shape[1]),
        'names': names,
        'bounds': bounds
    }
    print('compSol: ', compSol_full)
    for i in range(measured_outcomes.shape[1]):
        '''
        if (i == 0 or i == 1 or i == 2 or i == 4) and compSol_full == 'FB171' and rob_objs == 'objs':
            continue
        if (i == 5 or i == 7 or i == 9) and compSol_full == 'FB171' and rob_objs == 'objs':
            continue
        if (i == 10 or i == 11 or i == 12 or i == 14) and compSol_full == 'FB171' and rob_objs == 'objs':
            continue
        if (i == 15 or i == 17 or i == 19) and compSol_full == 'FB171' and rob_objs == 'objs':
            continue
        
        if (i == 0  or i == 1 or i == 2 or i == 4) and compSol_full == 'PW113' and rob_objs == 'objs':
            continue
        if (i == 5 or i == 7 or i == 9) and compSol_full == 'PW113' and rob_objs == 'objs':
            continue
        if (i == 10 or i == 11 or i == 12 or i == 14) and compSol_full == 'PW113' and rob_objs == 'objs':
            continue
        if (i == 15 or i == 17 or i == 19) and compSol_full == 'PW113' and rob_objs == 'objs':
            continue
        
        if (i == 0 or i == 1  or i == 2 or i == 4) and compSol_full == 'LS98' and rob_objs == 'objs':
            continue
        if (i == 5 or i == 7 or i == 9) and compSol_full == 'LS98' and rob_objs == 'objs':
            continue
        if (i == 10 or i == 11 or i == 12 or i == 14) and compSol_full == 'LS98' and rob_objs == 'objs':
            continue
        if (i == 15 or i == 17 or i == 19) and compSol_full == 'LS98' and rob_objs == 'objs':
            continue
        
        if i == 7 and compSol_full == 'PW113' and rob_objs == 'objs':
            continue
        
        # ignore the INF_F
        if i == 14 and compSol_full == 'LS98' and rob_objs == 'objs':
            continue
        if i == 2 and compSol_full == 'PW113' and rob_objs == 'objs':
            continue
        '''
        
        mo_label = mo_names[i]
        print('obj: ', mo_label)
        
        filename = 'delta_output/delta_' + rob_objs + '_' + mode + '/S1_' + mo_label + '_' + compSol_full + '.csv'
        
        #filename = 'delta_output/delta_base_rdm/S1_' + mo_label + '_' + compSol_full + '.csv'
        S1 = delta.analyze(problem, X, Y[mo_label].values, num_resamples=10, conf_level=0.95, print_to_console=False)
        numpy_S1 = np.array(S1["S1"])
        fileout = pd.DataFrame([names, numpy_S1], index = None, columns = None)
        fileout.to_csv(filename, sep=",")

'''
Name all file headers and compSol to be analyzed
'''
obj_names = ['REL_W', 'RF_W', 'INF_NPC_W', 'PFC_W', 'WCC_W', \
        'REL_D', 'RF_D', 'INF_NPC_D', 'PFC_D', 'WCC_D', \
        'REL_F', 'RF_F', 'INF_NPC_F', 'PFC_F', 'WCC_F', \
        'REL_R', 'RF_R', 'INF_NPC_R', 'PFC_R', 'WCC_R']

    
dv_names = ['RT_W', 'RT_D', 'RT_F', 'TT_D', 'TT_F', 'LMA_W', 'LMA_D', 'LMA_F',\
            'RC_W', 'RC_D', 'RC_F', 'IT_W', 'IT_D', 'IT_F', 'INF_W', 'INF_D', 'INF_F']
'''
DU factor keys:
    WRE: Watertown restriction efficiency
    DRE: Dryville restriction efficiency
    FRE: Fallsland restriction efficiency
    DMP: Demand multiplier
    BTM: Bond term multiplier
    BIM: Bond interest rate multiplier
    IIM: Infrastructure interest rate multiplier
    EMP: Evaporation rate multplier
    STM: Streamflow amplitude multiplier
    SFM: Streamflow frequency multiplier
    SPM: Streamflow phase multiplier
'''

rdm_headers_dmp = ['WRE', 'DRE', 'FRE']
rdm_headers_utilities = ['DMP', 'BTM', 'BIM', 'IIM']
rdm_headers_inflows = ['STM', 'SFM', 'SPM']
rdm_headers_ws = ['EMP', 'CRR PTD', 'CRR CTD', 'LM PTD', 'LM CTD', 'AL PTD', 
                  'AL CTD', 'D PTD', 'D CTD', 'NRR PTDD', 'NRR CTD', 'SCR PTD', 
                  'SCT CTD', 'GC PTD', 'GC CTD', 'CRR_L PT', 'CRR_L CT', 
                  'CRR_H PT', 'CRR_H CT', 'WR1 PT', 'WR1 CT', 'WR2 PT', 
                  'WR2 CT', 'DR PT', 'DR CT', 'FR PT', 'FR CT']

rdm_headers_ws_drop = ['CRR PTD', 'CRR CTD', 'LM PTD', 'LM CTD', 'AL PTD', 
                       'AL CTD', 'D PTD', 'D CTD', 'NRR PTDD', 'NRR CTD', 
                       'SCR PTD', 'SCT CTD', 'GC PTD', 'GC CTD']

rdm_all_headers = ['WRE', 'DRE', 'FRE', 'DMP', 'BTM', 'BIM', 'IIM', 
                   'STM', 'SFM', 'SPM', 'EMP', 'CRR_L PT', 'CRR_L CT', 
                   'CRR_H PT', 'CRR_H CT', 'WR1 PT', 'WR1 CT', 'WR2 PT', 
                   'WR2 CT', 'DR PT', 'DR CT', 'FR PT', 'FR CT']

utilities = ['Watertown', 'Dryville', 'Fallsland', 'Regional']

N_RDMS = 1000
N_SOLNS = 1000

# different DU scenarios
bad_scenario = 223   # evap multiplier = 1.2, demand multiplier = 1.95
optimistic_scenario = 782   # evap multiplier = 0.82, demand multiplier = 0.54
baseline_scenario = 229     # evap multiplier = 1.0, demand multiplier = 0.99

'''
Load DU factor files and DV files
'''
rdm_factors_directory = '/home/fs02/pmr82_0001/lbl59/Implementation_Uncertainty/WaterPaths_duReeval/TestFiles/'
rdm_dmp_filename = rdm_factors_directory + 'rdm_dmp_test_problem_reeval.csv'
rdm_utilities_filename = rdm_factors_directory + 'rdm_utilities_test_problem_reeval.csv'
rdm_inflows_filename = rdm_factors_directory + 'rdm_inflows_test_problem_reeval.csv'
rdm_watersources_filename = rdm_factors_directory + 'rdm_water_sources_test_problem_reeval.csv'

rdm_dmp = pd.read_csv(rdm_dmp_filename, sep=",", names=rdm_headers_dmp)
rdm_utilities = pd.read_csv(rdm_utilities_filename, sep=",", names=rdm_headers_utilities)
rdm_inflows = pd.read_csv(rdm_inflows_filename, sep=",", names=rdm_headers_inflows)
rdm_ws_full = np.loadtxt(rdm_watersources_filename, delimiter=",")
rdm_ws = pd.DataFrame(rdm_ws_full[:, :len(rdm_headers_ws)], columns=rdm_headers_ws)

rdm_ws = rdm_ws.drop(rdm_headers_ws_drop, axis=1)

dufs = pd.concat([rdm_dmp, rdm_utilities, rdm_inflows, rdm_ws], axis=1, ignore_index=True)
dufs.columns = rdm_all_headers
dufs_np = dufs.to_numpy()

duf_numpy = dufs_np[:1000, :]

dv_directory = '/home/fs02/pmr82_0001/lbl59/Implementation_Uncertainty/WaterPaths_duReeval/IU_Samples/'


'''
2 - Get bounds for DU factors 
'''
duf_bounds = find_bounds(duf_numpy)

'''
3 - Load robustness file and objectives file
'''

# to change
compSol_names = ['FB171', 'PW113', 'LS98']
compSol_names_short = ['FB', 'PW', 'LS']

for c in range(len(compSol_names)):
    compSol_full = compSol_names[c]
    compSol = compSol_names_short[c]
    
    '''
    2 - Get bounds for DVs
    '''  
    dv_filename = dv_directory + 'IU_allMeasures_' + compSol + '.csv'

    dvs = np.loadtxt(dv_filename, delimiter=",")
    dvs = np.delete(dvs, [14,15,16], 1)
    dv_bounds_filename = '/home/fs02/pmr82_0001/lbl59/Implementation_Uncertainty/'  + compSol + '_data/IU_ranges.txt'
    dv_bounds = np.loadtxt(dv_bounds_filename, delimiter=" ", usecols=(1,2))
    
    out_directory = '/home/fs02/pmr82_0001/lbl59/Implementation_Uncertainty/post_processing_du/DU_reeval_output_Apr2022/' + \
                    compSol_full + '/'
    '''
    Change here!
    '''
    dv_du = dvs[:1000, :len(dv_names)]  # Change depending on RDM factor being analyzed
    names = dv_names  # Change depending on whether DVs for DU factors are being analyzed
    bounds = dv_bounds  # Change depending on whether DVs for DU factors are being analyzed
    mode = 'DV'  # Change depending on whether DVs (DV) for DU factors (DUF) are being analyzed

    robustness_filename = out_directory + 'robustness_perturbed_og_' + compSol_full + '.csv'
    robustness_arr = np.loadtxt(robustness_filename, delimiter=",")
    robustness_df = pd.DataFrame(robustness_arr[:1000, :], columns=utilities)

    objs_filename = ""
    if mode == 'DV':
        #objs_filename = out_directory + 'meanObjs_acrossRDM_' + compSol_full + '.csv'
        objs_filename = '/home/fs02/pmr82_0001/lbl59/Implementation_Uncertainty/WaterPaths_duReeval/Objectives_' + \
            compSol + '_perturbed_Apr2022/Objectives_RDM' + str(baseline_scenario) + '_sols0_to_1000.csv'
    elif mode == 'DUF':
        objs_filename = out_directory + 'meanObjs_acrossSoln_' + compSol_full + '.csv'
    
    objs_arr = np.loadtxt(objs_filename, delimiter=",")
    objs_all = np.zeros((N_SOLNS,20), dtype=float)
    objs_all[:,:15] = objs_arr
    objs_all = minimax(N_SOLNS, objs_all)
    objs_df = pd.DataFrame(objs_all[:1000, :], columns=obj_names)

    '''
    Change here!
    '''
    measured_outcomes = objs_df  # Change depending on objs or robustness being analyzed
    mo_names = obj_names    # Change depending on objs or robustness being analyzed
    rob_objs = 'objs'   # Change depending on objs ('objs') or robustness ('robustness') being analyzed

    delta_sensitivity(dv_du, measured_outcomes, names, mo_names, bounds, compSol_full, rob_objs)
