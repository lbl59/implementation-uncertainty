# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 17:36:28 2022

@author: lbl59
"""
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
#import seaborn as sns
#sns.set_theme('whitegrid')

def calc_percent_deviation(dvs_perturbed_df, dvs_og_df):
    percent_deviation = np.zeros((dvs_perturbed_df.shape[0], dvs_perturbed_df.shape[1]), dtype=float)
    
    for i in range(dvs_perturbed_df.shape[1]):
        percent_deviation[:,i] = \
            (dvs_perturbed_df.iloc[:,i] - dvs_og_df.iloc[0,i])*100
    
    percent_deviation_df = pd.DataFrame(percent_deviation, columns=dv_names, index=None)
    return percent_deviation_df

def calc_percent_robustness_change(robustness_p, robustness_og):
    percent_change_W = (robustness_p['Watertown'] - robustness_og['Watertown'])/robustness_og['Watertown']
    percent_change_D = (robustness_p['Dryville'] - robustness_og['Dryville'])/robustness_og['Dryville']
    percent_change_F = (robustness_p['Fallsland'] - robustness_og['Fallsland'])/robustness_og['Fallsland']
    percent_change_R = (robustness_p['Regional'] - robustness_og['Regional'])/robustness_og['Regional']
    
    return percent_change_W, percent_change_D, percent_change_F, percent_change_R

def robustness_change(robustness_p, robustness_og):
    high_low_W = np.array(robustness_p['Watertown'] >= robustness_og['Watertown']).astype(int)
    high_low_D = np.array(robustness_p['Dryville'] >= robustness_og['Dryville']).astype(int)
    high_low_F = np.array(robustness_p['Fallsland'] >= robustness_og['Fallsland']).astype(int)
    high_low_R = np.array(robustness_p['Regional'] >= robustness_og['Regional']).astype(int)
    
    return high_low_W, high_low_D, high_low_F, high_low_R

def minimax(N_SOLNS, objs):
    for i in range(N_SOLNS):
        for j in range(5):
            if j == 0:
                objs[i,15] = np.min([objs[i,0],objs[i,5], objs[i,10]])
            else:
                objs[i, (j+15)] = np.max([objs[i,j],objs[i,j+5], objs[i,j+10]])
    return objs

'''
Change these!!!
'''
compSol = 'PW'
compSol_full = 'PW113'
compSol_num = 2

compSol_labels = ['Fallback bargaining', 'Least-squares', 'Power index']
compSol_cmaps = ['YlOrBr', 'Purples', 'Greens']
compSol_colors = ['darkorange', 'purple', 'forestgreen']
worst_robustness = [543, 433, 250]
best_robustness = [564, 589, 4]

compSol_label = compSol_labels[compSol_num]
compSol_cmap = compSol_cmaps[compSol_num]
compSol_col = compSol_colors[compSol_num]

regional_worst_idx = worst_robustness[compSol_num]
regional_best_idx = best_robustness[compSol_num]


obj_names = ['REL_W', 'RF_W', 'INF_NPC_W', 'PFC_W', 'WCC_W', \
        'REL_D', 'RF_D', 'INF_NPC_D', 'PFC_D', 'WCC_D', \
        'REL_F', 'RF_F', 'INF_NPC_F', 'PFC_F', 'WCC_F', \
        'REL_R', 'RF_R', 'INF_NPC_R', 'PFC_R', 'WCC_R']

dv_names = ['RT_W', 'RT_D', 'RT_F', 'TT_D', 'TT_F', 'LMA_W', 'LMA_D', 'LMA_F',\
            'RC_W', 'RC_D', 'RC_F', 'IT_W', 'IT_D', 'IT_F', 'IP_W', 'IP_D', \
            'IP_F', 'INF_W', 'INF_D', 'INF_F']
    
utilities = ['Watertown', 'Dryville', 'Fallsland', 'Regional']
utils_indv = ['Watertown', 'Dryville', 'Fallsland']

'''
Change here
'''
util = 'Regional'
util_num = 3    # Watertown is 0, Dryville is 1, Fallsland is 2
x_key = 'DMP'
x_lab = 'Demand multiplier'
y_key = 'RF_R'
y_lab = util +  ' restr. freq (%) $\longrightarrow$'
z_key = 'WCC_R'
z_lab = util +  ' worst-case cost (%) $\longrightarrow$'
size_key = 'INF_NPC_R'
color_key = 'DMP'
color_key = 'REL_R'

bad_scenario = 223   # evap multiplier = 1.2, demand multiplier = 1.95
optimistic_scenario = 782   # evap multiplier = 0.82, demand multiplier = 0.54
baseline_scenario = 229     # evap multiplier = 1.0, demand multiplier = 0.99

chosen_scenario = baseline_scenario
savefig_name = 'base'
title_ext = '\nBehavior across baseline scenario'

'''
Get and process DV data
'''
dir_dvs = '/home/fs02/pmr82_0001/lbl59/Implementation_Uncertainty/WaterPaths_duReeval/IU_Samples/'
filename_dvs_perturbed = 'IU_allMeasures_' + compSol + '.csv'
filename_dvs_og = compSol +  '_soln.csv'

dvs_perturbed_df = pd.read_csv(dir_dvs + filename_dvs_perturbed, sep=',', names=dv_names, index_col=False)
dvs_perturbed_df = dvs_perturbed_df.drop(['IP_W', 'IP_D', 'IP_F'], axis=1)

dvs_og_df = pd.read_csv(dir_dvs + filename_dvs_og, sep=',', names=dv_names, index_col=False)
dvs_og_df = dvs_og_df.drop(['IP_W', 'IP_D', 'IP_F'], axis=1)

dvs_all_df = pd.concat([dvs_perturbed_df, dvs_og_df], axis=0)

'''
Get objective values data
'''
objs_directory = '/home/fs02/pmr82_0001/lbl59/Implementation_Uncertainty/WaterPaths_duReeval/Objectives_' + \
    compSol + '_perturbed_Apr2022/Objectives_RDM' + str(chosen_scenario) + \
    '_sols0_to_1000.csv'

df_objs = pd.read_csv(objs_directory, index_col=False, names = obj_names)
df_objs_arr = np.zeros((1000, 20), dtype=float)
df_objs_arr[:,:15] = np.loadtxt(objs_directory, delimiter=',')
df_objs_arr = minimax(1000, df_objs_arr)
df_objs = pd.DataFrame(df_objs_arr, columns=obj_names)

'''
Get RDM data
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

directory = '/home/fs02/pmr82_0001/lbl59/Implementation_Uncertainty/'+\
    'post_processing_du/DU_reeval_output_Apr2022/' + compSol_full + '/'

# change depending on compromise solution being analyzed
filename_robustness = 'robustness_perturbed_og_' + compSol_full + '.csv'
robustness_df = pd.read_csv(directory + filename_robustness, sep=',', names=utilities)

x_data = (dufs.loc[:999][x_key].values)
y_data = (df_objs[y_key].values)*100
z_data = (df_objs[z_key].values)*100
size_data = (df_objs[size_key].values)*50

fig = plt.figure(figsize=(10,10))

cmap = matplotlib.cm.get_cmap(compSol_cmap)

ax = fig.add_subplot(projection='3d')

ax.scatter3D(x_data, y_data, z_data, s=20, c=df_objs[color_key].values, cmap=cmap,  
             alpha=0.8)


ax.set_xlabel(x_lab, labelpad=10)

ax.set_ylabel(y_lab, labelpad=10)

ax.set_zlabel(z_lab, labelpad=10)

# change depending on objective you want to plot
title = compSol_label + title_ext


robustness_savefile = '/home/fs02/pmr82_0001/lbl59/Implementation_Uncertainty/' + \
    'post_processing_du/Figures/3d_plots/' + util +  savefig_name + '_' + compSol_full + '_' + '_satisficing.pdf'

plt.legend(bbox_to_anchor=(0.3, 1.0), loc='upper right', prop={'size': 9}, markerscale=0.5, ncol=1)
plt.title(title, size=12)
plt.savefig(robustness_savefile)
plt.show()
