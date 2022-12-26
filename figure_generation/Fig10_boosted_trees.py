# -*- coding: utf-8 -*-
"""
Created on Thu May 12 22:29:04 2022

@author: lbl59
"""
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from copy import deepcopy
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
sns.set(style='white')


'''
Function to check whether performance criteria are met
'''
def check_satisficing(objs, objs_col, satisficing_bounds):
    meet_low = objs[:, objs_col] >= satisficing_bounds[0]
    meet_high = objs[:, objs_col] <= satisficing_bounds[1]

    meets_criteria = np.hstack((meet_low, meet_high)).all(axis=1)

    return meets_criteria

'''
Function to perform regional minimax
'''
def minimax(N_SOLNS, objs):
    for i in range(N_SOLNS):
        for j in range(5):
            if j == 0:
                objs[i,15] = np.min([objs[i,0],objs[i,5], objs[i,10]])
            else:
                objs[i, (j+15)] = np.max([objs[i,j],objs[i,j+5], objs[i,j+10]])
    return objs


'''
0 - Name all file headers and compSol to be analyzed
'''
obj_names = ['REL_W', 'RF_W', 'INF_NPC_W', 'PFC_W', 'WCC_W', \
        'REL_D', 'RF_D', 'INF_NPC_D', 'PFC_D', 'WCC_D', \
        'REL_F', 'RF_F', 'INF_NPC_F', 'PFC_F', 'WCC_F', \
        'REL_R', 'RF_R', 'INF_NPC_R', 'PFC_R', 'WCC_R']

dv_names = ['RT_W', 'RT_D', 'RT_F', 'TT_D', 'TT_F', 'LMA_W', 'LMA_D', 'LMA_F',\
            'RC_W', 'RC_D', 'RC_F', 'IT_W', 'IT_D', 'IT_F', 'INF_W', 'INF_D', 'INF_F']

'''
Keys:
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

all_headers = ['RT_W', 'RT_D', 'RT_F', 'TT_D', 'TT_F', 'LMA_W', 'LMA_D', 'LMA_F',
               'RC_W', 'RC_D', 'RC_F', 'IT_W', 'IT_D', 'IT_F', 'INF_W', 'INF_D', 'INF_F',
               'WRE', 'DRE', 'FRE', 'DMP', 'BTM', 'BIM', 'IIM', 
               'STM', 'SFM', 'SPM', 'EMP', 'CRR_L PT', 'CRR_L CT', 
               'CRR_H PT', 'CRR_H CT', 'WR1 PT', 'WR1 CT', 'WR2 PT', 
               'WR2 CT', 'DR PT', 'DR CT', 'FR PT', 'FR CT']

utilities = ['Watertown', 'Dryville', 'Fallsland', 'Regional']

N_RDMS = 1000
N_SOLNS = 1000

compSol = 'PW'  # Change depending on compSol being analyzed
compSol_full = 'PW113'  # Change depending on compSol being analyzed
compSol_num = 2   # FB is 0, LS is 1, PW is 2

worst_robustness_W = [610, 91, 260]
worst_robustness_D = [543, 433, 250]
worst_robustness_F = [143, 81, 337]
worst_robustness_R = [543, 433, 250]

#best_robustness = [564, 589, 4]

bad_scenario = 223   # evap multiplier = 1.2, demand multiplier = 1.95
#optimistic_scenario = 782   # evap multiplier = 0.82, demand multiplier = 0.54
baseline_scenario = 229     # evap multiplier = 1.0, demand multiplier = 0.99

worst_rdm_W = worst_robustness_W[compSol_num]   
worst_rdm_D = worst_robustness_D[compSol_num]   
worst_rdm_F = worst_robustness_F[compSol_num]   
worst_rdm_R = worst_robustness_R[compSol_num]   

#best_rdm = best_robustness[compSol_num]     

'''
1 - Load DU factor files 
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

'''
Load DV files
'''
dv_directory = '/home/fs02/pmr82_0001/lbl59/Implementation_Uncertainty/WaterPaths_duReeval/IU_Samples/'
dv_pt_filename = dv_directory + 'IU_allMeasures_' + compSol + '.csv'
dvs_pt_arr_full = np.loadtxt(dv_pt_filename, delimiter=",")
dvs_pt_arr_full = np.delete(dvs_pt_arr_full, [14,15,16], 1)
dvs_pt_arr = dvs_pt_arr_full[:, :len(dv_names)]

all_params = np.concatenate([dvs_pt_arr, duf_numpy], axis=1)

'''
3 - Load objectives and robustness files
'''
out_directory = '/home/fs02/pmr82_0001/lbl59/Implementation_Uncertainty/post_processing_du/DU_reeval_output_Apr2022/' + \
                compSol_full + '/'

# objective values across all RDMs for the original solution
objs_filename_og = out_directory + 'original_compromise_acrossSoln_' + compSol_full + '.csv'
objs_arr_og = np.loadtxt(objs_filename_og, delimiter=",")
objs_df_og = pd.DataFrame(objs_arr_og[:1000, :], columns=obj_names)

# objective values across all RDMs for the solution with the worst regional robustness
# Watertown
objs_worstRobustness_filename_W = out_directory + 'worst_robustness_objs_acrossRDMs_W_' + \
    compSol_full + '.csv'
objs_df_worst_W = pd.read_csv(objs_worstRobustness_filename_W, header=0)

# Dryville
objs_worstRobustness_filename_D = out_directory + 'worst_robustness_objs_acrossRDMs_D_' + \
    compSol_full + '.csv'
objs_df_worst_D = pd.read_csv(objs_worstRobustness_filename_D, header=0)

# Fallsland
objs_worstRobustness_filename_F = out_directory + 'worst_robustness_objs_acrossRDMs_F_' + \
    compSol_full + '.csv'
objs_df_worst_F = pd.read_csv(objs_worstRobustness_filename_F, header=0)

# Regional
objs_worstRobustness_filename_R = out_directory + 'worst_robustness_objs_acrossRDMs_R_' + \
    compSol_full + '.csv'
objs_df_worst_R = pd.read_csv(objs_worstRobustness_filename_R, header=0)

'''
# objective values across all RDMs for the solution with the best regional robustness
objs_bestRobustness_filename = out_directory + 'best_robustness_objs_acrossRDMs_soln' + str(best_rdm) + '_' + \
    compSol_full + '.csv'
objs_df_best = pd.read_csv(objs_bestRobustness_filename, header=0)
'''

# robustness of each solution across all RDMs
robustness_filename = out_directory + 'robustness_perturbed_og_' + compSol_full + '.csv'
robustness_arr = np.loadtxt(robustness_filename, delimiter=",")
robustness_df = pd.DataFrame(robustness_arr[:1000, :], columns=utilities)
robustness_og = pd.DataFrame(np.reshape(robustness_arr[1000, :],(1,4)), columns=utilities)

'''
4 - Determine the type of factor maps to plot. Plot for:
    'allParams_og'
    'allParams_best'
    'allParams_worst'
    for both 'objs' and 'robustness'
'''
mode_fignames = ['gbt_og', 'gbt_worst_W', 'gbt_worst_D', 'gbt_worst_F', 'gbt_worst_R']
mode = 'objs_satisficing'
objs_dfs = [objs_df_og, objs_df_worst_W, objs_df_worst_D, objs_df_worst_F, objs_df_worst_R]
factor_names = all_headers

for i in range(5):
    
    mode_figname = mode_fignames[i] # do worst, best and original
    objs_df = objs_dfs[i]
    
    '''
    5 - Extract each utility's set of performance objectives and robustness 
    '''
    Watertown_all = objs_df[['REL_W', 'RF_W', 'INF_NPC_W', 'PFC_W', 'WCC_W']].to_numpy()
    Dryville_all = objs_df[['REL_D', 'RF_D', 'INF_NPC_D', 'PFC_D', 'WCC_D']].to_numpy()
    Fallsland_all = objs_df[['REL_F', 'RF_F', 'INF_NPC_F', 'PFC_F', 'WCC_F']].to_numpy()
    Regional_all = objs_df[['REL_R', 'RF_R', 'INF_NPC_R', 'PFC_R', 'WCC_R']].to_numpy()
    
    # Watertown satisficing criteria
    rel_W = check_satisficing(Watertown_all, [0], [0.98, 1.0])
    rf_W = check_satisficing(Watertown_all, [1], [0.0, 0.1])
    wcc_W = check_satisficing(Watertown_all, [4], [0.0, 0.1])
    satisficing_W = rel_W*rf_W*wcc_W
    print('satisficing_W: ', satisficing_W.sum())
    
    # Dryville satisficing criteria
    rel_D = check_satisficing(Dryville_all, [0], [0.98, 1.0])
    rf_D = check_satisficing(Dryville_all, [1], [0.0, 0.1])
    wcc_D = check_satisficing(Dryville_all, [4], [0.0, 0.1])
    satisficing_D = rel_D*rf_D*wcc_D
    print('satisficing_D: ', satisficing_D.sum())
    
    # Fallsland satisficing criteria
    rel_F = check_satisficing(Fallsland_all, [0], [0.98,1.0])
    rf_F = check_satisficing(Fallsland_all, [1], [0.0,0.1])
    wcc_F = check_satisficing(Fallsland_all, [4], [0.0,0.1])
    satisficing_F = rel_F*rf_F*wcc_F
    print('satisficing_F: ', satisficing_F.sum())
    
    # Regional satisficing criteria
    rel_R = check_satisficing(Regional_all, [0], [0.98, 1.0])
    rf_R = check_satisficing(Regional_all, [1], [0.0, 0.1])
    wcc_R = check_satisficing(Regional_all, [4], [0.0, 0.1])
    satisficing_R = rel_R*rf_R*wcc_R
    print('satisficing_R: ', satisficing_R.sum())
    
    satisficing_dict = {'satisficing_W': satisficing_W, 'satisficing_D': satisficing_D,
                        'satisficing_F': satisficing_F, 'satisficing_R': satisficing_R}
    
    utils = ['satisficing_W', 'satisficing_D', 'satisficing_F', 'satisficing_R']
    
    '''
    6 - Fit a boosted tree classifier and extract important features
    '''
    for j in range(len(utils)):
        gbc = GradientBoostingClassifier(n_estimators=200,
                                         learning_rate=0.1,
                                         max_depth=3)
        
        # fit the classifier to each utility's sd
        # change depending on utility being analyzed!!
        criteria_analyzed = utils[j]
        df_to_fit = satisficing_dict[criteria_analyzed]
        gbc.fit(all_params, df_to_fit)
        
        feature_ranking = deepcopy(gbc.feature_importances_)
        feature_ranking_sorted_idx = np.argsort(feature_ranking)
        feature_names_sorted = [factor_names[i] for i in feature_ranking_sorted_idx]
        
        feature1_idx = len(feature_names_sorted) - 1
        feature2_idx = len(feature_names_sorted) - 2
        
        feature_figpath = '/home/fs02/pmr82_0001/lbl59/Implementation_Uncertainty/post_processing_du/' + \
            'Figures/scenario_discovery/Boosted_Trees/feature_ranking/' 
        print(feature_ranking_sorted_idx)
        
        feature_figname = feature_figpath + compSol_full + '_' + mode + '_' + criteria_analyzed + '.png'
        
        fig = plt.figure(figsize=(8, 8))
        ax = fig.gca()
        ax.barh(np.arange(len(feature_ranking)), feature_ranking[feature_ranking_sorted_idx])
        ax.set_yticks(np.arange(len(feature_ranking)))
        ax.set_yticklabels(feature_names_sorted)
        ax.set_xlim([0, 1])
        ax.set_xlabel('Feature ranking')
        plt.tight_layout()
        plt.savefig(feature_figname)
        
        '''
        7 - Create factor maps
        '''
        # select top 2 factors
        fm_figpath = '/home/fs02/pmr82_0001/lbl59/Implementation_Uncertainty/post_processing_du/' + \
            'Figures/scenario_discovery/Boosted_Trees/' + compSol_full + '_' + mode + '/' 
        fm_figname = fm_figpath + 'factor_map_' + mode_figname + '_' + criteria_analyzed + '.pdf'
        
        selected_factors = all_params[:, [feature_ranking_sorted_idx[feature1_idx], 
                                         feature_ranking_sorted_idx[feature2_idx]]]
        gbc_2_factors = GradientBoostingClassifier(n_estimators=200,
                                                   learning_rate=0.1,
                                                   max_depth=3)
        
        # change this one depending on utility and compSol being analyzed
        gbc_2_factors.fit(selected_factors, df_to_fit)
        
        x_data = selected_factors[:, 0]
        y_data = selected_factors[:, 1]
        
        x_min, x_max = (x_data.min(), x_data.max())
        y_min, y_max = (y_data.min(), y_data.max())
        
        xx, yy = np.meshgrid(np.arange(x_min, x_max*1.001, (x_max-x_min)/100),
                             np.arange(y_min, y_max*1.001, (y_max-y_min)/100))
        
        dummy_points = list(zip(xx.ravel(), yy.ravel()))
        
        z = gbc_2_factors.predict_proba(dummy_points)[:,1]
        z[z<0] = 0
        z = z.reshape(xx.shape)
        
        fig_factormap = plt.figure(figsize = (10,8))
        ax_f = fig_factormap.gca()
        ax_f.contourf(xx, yy, z, [0, 0.5, 1], cmap='RdBu', alpha=0.6, vmin=0, vmax=1)
        
        # change robustness here!
        ax_f.scatter(selected_factors[:,0], selected_factors[:,1],
                     c=df_to_fit, cmap='Reds_r', edgecolor='grey',
                     alpha=0.6, s=100, linewidths=0.5)
        
        # indicate drought + high demand scenario
        ax_f.scatter(selected_factors[bad_scenario,0], selected_factors[bad_scenario,1],
                     c='k', edgecolors='k', marker = 'X', alpha=1.0, s=450, linewidths=4, label='Challenging SOW')
        # indicate baseline scenario
        ax_f.scatter(selected_factors[baseline_scenario,0], selected_factors[baseline_scenario,1],
                     c='k', edgecolors='k', marker = '^', alpha=1.0, s=450, linewidths=4, label='Baseline SOW')
        '''
        # indicate wet + low demand scenario
        ax_f.scatter(selected_factors[optimistic_scenario,0], selected_factors[optimistic_scenario,1],
                     c='k', edgecolors='k', marker = '*', alpha=1.0, s=550, linewidths=4, label='Optimistic SOW')
        '''
        #ax_f.set_xlim([1.0, 1.2])
        #ax_f.set_ylim([1.0, 1.2])
        ax_f.set_xlabel(factor_names[feature_ranking_sorted_idx[feature1_idx]], size=16)
        ax_f.set_ylabel(factor_names[feature_ranking_sorted_idx[feature2_idx]], size=16)
        ax_f.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08),
                    fancybox=True, shadow=True, ncol=3, markerscale=0.5)

        plt.savefig(fm_figname)