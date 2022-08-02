# -*- coding: utf-8 -*-
"""
Created on Mon Apr  18 17:36:28 2022

Conducts logistic regression on the robustness of the perturbed isntances of each 
compromise solution. 

@author: lbl59
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from logisticRegressionTools import fitAllLogit
from logisticRegressionTools import plotCombinedFactorMaps_SOS
import warnings
from pandas.core.common import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

def check_satisficing(objs, objs_col, satisficing_bounds):
    """
    Identifies if the robustness of one perturbed instance meets or exceeds the
    original robustness value.

    Parameters
    ----------
    objs : numpy array
        A numpy array of floats containing the robustness values across all
        perturbed instances
    objs_col : int
        Integer indicators for each utility where 0 is Watertown, 1 is Dryville, 
        2 is Fallsland and 3 is Regional.
    satisficing_bounds : list
        A length-2 list indicating the lower (original value) and upper (100%) bound
        of robustness.

    Returns
    -------
    meets_criteria : boolean
        Returns true if the current perturbed instance results in a robustness value 
        that is at least as good as the original robustness. Returns false otherwise.

    """
    
    meet_low = objs[:, objs_col] >= satisficing_bounds[0]
    meet_high = objs[:, objs_col] <= satisficing_bounds[1]

    meets_criteria = np.hstack((meet_low, meet_high)).all(axis=1)

    return meets_criteria


'''
0 - Name all file headers and compSol to be analyzed
'''
rdm_headers_dmp = ['WRE', 'DRE', 'FRE']
rdm_headers_utilities = ['DMP', 'BTM', 'BIM', 'IIM']
rdm_headers_inflows = ['STM', 'SFM', 'SPM']
rdm_all_headers = ['WRE', 'DRE', 'FRE', 'DMP']

dv_names = ['RT_W', 'RT_D', 'RT_F', 'TT_D', 'TT_F', 'LMA_W', 'LMA_D', 'LMA_F',
            'RC_W', 'RC_D', 'RC_F', 'IT_W', 'IT_D', 'IT_F', 'INF_W', 'INF_D', 'INF_F']

utilities = ['Watertown', 'Dryville', 'Fallsland', 'Regional']

# Only DMP is used as an indicator out of all the other DU Factors as it was previously 
# found using Boosted Trees to have the strongest influence over a utility's vulnerability.
all_headers = ['RT_W', 'RT_D', 'RT_F', 'TT_D', 'TT_F', 'LMA_W', 'LMA_D', 'LMA_F',
               'RC_W', 'RC_D', 'RC_F', 'IT_W', 'IT_D', 'IT_F', 'INF_W', 'INF_D', 
               'INF_F', 'DMP']

'''
MAKE CHANGES HERE!
1 - Specify the following:
    Compromise solution 
    Utility for the given compromise solution
    Number of action triggers for selected utility
    Colormap
'''

compSol = 'PW'  # Change depending on compSol being analyzed
compSol_full = 'PW113'  # Change depending on compSol being analyzed
compSol_num = 2   # FB is 0, LS is 1, PW is 2

# indices of the perturbed instance with the worst robustness across the FB, LS 
# and PW compromise solutions
worst_robustness_W = [610, 91, 260]
worst_robustness_D = [543, 433, 250]
worst_robustness_F = [143, 81, 337]
worst_robustness_R = [543, 433, 250]

worst_W = worst_robustness_W[compSol_num]
worst_D = worst_robustness_D[compSol_num]
worst_F = worst_robustness_F[compSol_num]
worst_R = worst_robustness_R[compSol_num]

cmap_col = 'RdBu'

util = 'Watertown'
utils_short = 'W'

# params to choose determined using Delta Moment-Independent SA
# do this one at a time for each utility
param_i = 'DMP'
param_j = 'RT_W'
worst = worst_W

'''
2 - Load DV files
'''
dir_dvs = '/home/fs02/pmr82_0001/lbl59/Implementation_Uncertainty/WaterPaths_duReeval/IU_Samples/'
filename_dvs_perturbed = dir_dvs + 'IU_allMeasures_' + compSol + '.csv'
filename_dvs_og = dir_dvs + compSol +  '_soln.csv'

# load decision variables
dvs_pt_arr_full = (np.loadtxt(filename_dvs_perturbed, delimiter=","))*100
dvs_pt_arr_full = np.delete(dvs_pt_arr_full, [14,15,16], 1) # delete the IP decision variable values
dvs_pt_arr = dvs_pt_arr_full[:, :len(dv_names)]
dvs_perturbed_df = pd.DataFrame(dvs_pt_arr, columns=dv_names)

dvs_og_arr_full = (np.loadtxt(filename_dvs_og, delimiter=","))*100
dvs_og_arr_full = np.delete(dvs_og_arr_full, [14,15,16]) # delete the IP decision variable values
dvs_og_arr = dvs_og_arr_full[:len(dv_names)]
dvs_og_df = pd.DataFrame(np.reshape(dvs_og_arr,(1,17)), columns=dv_names)

'''
# normalize DV values
DV_range_dir = '/home/fs02/pmr82_0001/lbl59/Implementation_Uncertainty/' + compSol + '_data/'
DVparamBounds = np.genfromtxt(DV_range_dir + 'IU_ranges.txt', delimiter =' ', )
'''

'''
3 - Load DU factor files 
'''
rdm_factors_directory = '/home/fs02/pmr82_0001/lbl59/Implementation_Uncertainty/WaterPaths_duReeval/TestFiles/'
rdm_dmp_filename = rdm_factors_directory + 'rdm_dmp_test_problem_reeval.csv'
rdm_utilities_filename = rdm_factors_directory + 'rdm_utilities_test_problem_reeval.csv'
rdm_inflows_filename = rdm_factors_directory + 'rdm_inflows_test_problem_reeval.csv'
rdm_watersources_filename = rdm_factors_directory + 'rdm_water_sources_test_problem_reeval.csv'

rdm_utilities = pd.read_csv(rdm_utilities_filename, sep=",", names=rdm_headers_utilities)

dufs = rdm_utilities.drop(['BTM', 'BIM', 'IIM'], axis=1)
dufs.columns = ['DMP']
dufs_np = dufs.to_numpy()

duf_numpy = dufs_np[:1000, :]

'''
4 - Wrangle full DU factor and DV array
'''
all_params = np.concatenate([dvs_pt_arr, duf_numpy], axis=1)
all_params_df = pd.DataFrame(all_params, columns=all_headers)

# add intercept column
all_params_df['intercept'] = np.ones(np.shape(all_params_df)[0])

'''
3 - Load robustness files
'''
out_directory = '/home/fs02/pmr82_0001/lbl59/Implementation_Uncertainty/post_processing_du/DU_reeval_output_Apr2022/' + \
                compSol_full + '/'

# robustness of each solution across all RDMs
robustness_filename = out_directory + 'robustness_perturbed_og_' + compSol_full + '.csv'
robustness_arr = np.loadtxt(robustness_filename, delimiter=",")
robustness_pt = pd.DataFrame(robustness_arr[:1000, :], columns=utilities)
robustness_og = pd.DataFrame(np.reshape(robustness_arr[1000, :],(1,4)), columns=utilities)

robustness_og_W = robustness_og[['Watertown']].values
robustness_og_D = robustness_og[['Dryville']].values
robustness_og_F = robustness_og[['Fallsland']].values
robustness_og_R = robustness_og[['Regional']].values

robustness_W = check_satisficing(robustness_pt.to_numpy(), [0], [robustness_og_W,1.0])
robustness_D = check_satisficing(robustness_pt.to_numpy(), [1], [robustness_og_D,1.0])
robustness_F = check_satisficing(robustness_pt.to_numpy(), [2], [robustness_og_F,1.0])
robustness_R = check_satisficing(robustness_pt.to_numpy(), [3], [robustness_og_R,1.0])


satisficing_dict = {'W': robustness_W, 'D': robustness_D, 'F': robustness_F, 'R': robustness_R}
satisficing_df = pd.DataFrame(satisficing_dict)

'''
4 - Set robustness criteria
Based on boosted trees, select the two criteria that most affect the utility's ability to meet 
or exceed the original regional robustness

Criteria for safe operating space: 
    The region of the decision space such that any combination of decision variables sampled within
    the region is at least as robust original regional robustness of the original compromise
    water portfolio.
'''
# Regional 2D
fig = plt.figure()
sns.set()
levels = np.linspace(0,1.00, 201)

ax = fig.add_subplot(1,1,1)
result = fitAllLogit(robustness_pt, satisficing_dict[utils_short], 
                     all_params_df[['intercept', param_i, param_j]])

DVPredictors2 = ['RT_W', 'RT_D', 'RT_F', 'TT_D', 'TT_F', 'LMA_W', 'LMA_D', 'LMA_F',
               'RC_W', 'RC_D', 'RC_F', 'IT_W', 'IT_D', 'IT_F', 'INF_W', 'INF_D', 
               'INF_F', 'DMP']

LHBase = np.ones([1,len(DVPredictors2)])*.5
DV_base = pd.DataFrame(data=LHBase, columns=DVPredictors2)

# !!!change here!!!
min_x = min(all_params_df[param_i])
min_y = min(all_params_df[param_j])
max_x = max(all_params_df[param_i])
max_y = max(all_params_df[param_j])
    
param1_og = 1.0
param2_og = dvs_og_df[param_j].values
param1_w = all_params_df.loc[worst, param_i]
param2_w = dvs_perturbed_df.loc[worst, param_j]

# DMP
x_lim1 = 0.5
x_lim2 = 2.0

# relevant DV value
y_lim1 = param2_og - 4
if y_lim1 < 0:
    y_lim1 = 0
y_lim2 = param2_og + 4

plotCombinedFactorMaps_SOS(ax, result, util, param_i, param_j, x_lim1, x_lim2, y_lim1, y_lim2, 
                           param1_og, param2_og, param1_w, param2_w, [], [], levels)

ax.legend(loc=1, prop={'size': 8}, markerscale=0.5)

# !!!change here!!!
savefig_path = '/home/fs02/pmr82_0001/lbl59/Implementation_Uncertainty/post_processing_du/Figures/' + \
    'scenario_discovery/Logistic_Regression/' + 'SOS/' + compSol + '_indv_' + utils_short + '.pdf'
plt.savefig(savefig_path)
plt.show()

