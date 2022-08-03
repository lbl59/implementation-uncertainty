"""
Created on Tue April 26 2022 16:12

@author: Lillian Bei Jia Lau

Gathers the delta sensitivity indices into files per utility
"""

import numpy as np
import pandas as pd

'''
Name all utilities, objectives, DVs and DU factors
'''
utilities = ['_W', '_D', '_F', '_R']
objs = ['REL', 'RF', 'INF_NPC', 'PFC', 'WCC']
objs_alt = ['REL', 'RF', 'PFC', 'WCC']

utilities_full = ['Watertown', 'Dryville', 'Fallsland', 'Regional']

dv_names = ['RT_W', 'RT_D', 'RT_F', 'TT_D', 'TT_F', 'LMA_W', 'LMA_D', 'LMA_F',
            'RC_W', 'RC_D', 'RC_F', 'IT_W', 'IT_D', 'IT_F', 'INF_W', 'INF_D', 'INF_F']

duf_names = ['WRE', 'DRE', 'FRE', 'DMP', 'BTM', 'BIM', 'IIM', 
             'STM', 'SFM', 'SPM', 'EMP', 'CRR_L PT', 'CRR_L CT', 
             'CRR_H PT', 'CRR_H CT', 'WR1 PT', 'WR1 CT', 'WR2 PT', 
             'WR2 CT', 'DR PT', 'DR CT', 'FR PT', 'FR CT']

'''
Initialize numpy matrices and their associated dicts to store sensitivity values
'''
s1_dv_objs_W = np.zeros((len(objs), len(dv_names)), dtype=float)
s1_dv_objs_D = np.zeros((len(objs), len(dv_names)), dtype=float)
s1_dv_objs_F = np.zeros((len(objs), len(dv_names)), dtype=float)
s1_dv_objs_R = np.zeros((len(objs), len(dv_names)), dtype=float)

s1_dv_objs_dict = {
    '_W': s1_dv_objs_W,
    '_D': s1_dv_objs_D,
    '_F': s1_dv_objs_F,
    '_R': s1_dv_objs_R
}

s1_duf_objs_W = np.zeros((len(objs), len(duf_names)), dtype=float)
s1_duf_objs_D = np.zeros((len(objs), len(duf_names)), dtype=float)
s1_duf_objs_F = np.zeros((len(objs), len(duf_names)), dtype=float)
s1_duf_objs_R = np.zeros((len(objs), len(duf_names)), dtype=float)

s1_duf_objs_dict = {
    '_W': s1_duf_objs_W,
    '_D': s1_duf_objs_D,
    '_F': s1_duf_objs_F,
    '_R': s1_duf_objs_R
}

s1_dv_rob = np.zeros((len(utilities_full), len(dv_names)), dtype=float)
s1_duf_rob = np.zeros((len(utilities_full), len(duf_names)), dtype=float)

'''
Load in values
'''
compSol_names = ['FB171', 'PW113', 'LS98']
modes = ['DV', 'DUF']
#modes = ['DV']
rob_objs_list = ['robustness', 'objs']
#rob_objs_list = ['objs']
for c in range(len(compSol_names)):
    for m in range(len(modes)):
        for r in range(len(rob_objs_list)):
            compSol_full = compSol_names[c]
            mode = modes[m]
            rob_obj = rob_objs_list[r]
            
            main_dir = '/home/fs02/pmr82_0001/lbl59/Implementation_Uncertainty/post_processing_du/delta_output/delta_' + \
                       rob_obj + '_' + mode + '/'
            '''
            main_dir = '/home/fs02/pmr82_0001/lbl59/Implementation_Uncertainty/post_processing_du/delta_output/' + \
                'delta_base_rdm/'
            '''
            s1_util = []
            hdrs = []

            if rob_obj == 'robustness':
                if mode == 'DV':
                    hdrs = dv_names
                    s1_util = s1_dv_rob
                elif mode == 'DUF':
                    hdrs = duf_names
                    s1_util = s1_duf_rob

                for u in range(len(utilities_full)):
                    
                    curr_file = main_dir + 'S1_' + utilities_full[u] + '_' + compSol_full + '.csv'
                    s1_util[u, :] = (pd.read_csv(curr_file, sep=',', skiprows=2, header=None).iloc[0, 1:]).T

                s1_util_df = pd.DataFrame(s1_util, columns=hdrs)
                out_filepath = main_dir + 'all_utils_robustness_' + compSol_full + '.csv'

                s1_util_df.to_csv(out_filepath, sep=',', index=False)

            elif rob_obj == 'objs':
                for u in range(len(utilities_full)):
                    if mode == 'DV':
                        hdrs = dv_names
                        s1_util = s1_dv_objs_dict[utilities[u]]
                    elif mode == 'DUF':
                        hdrs = duf_names
                        s1_util = s1_duf_objs_dict[utilities[u]]

                    for j in range(len(objs)):
                        '''
                        if compSol_full == 'FB171' and j == 2:
                            continue
                        '''
                        if compSol_full == 'LS98' and j == 4:
                            continue
                        if compSol_full == 'PW113' and j == 2:
                            continue
                        
                        '''
                        if (j == 0 or j == 1 or j == 2 or j == 4) and compSol_full == 'FB171':
                            continue
                        if (j == 5 or j == 7 or j == 9) and compSol_full == 'FB171':
                            continue
                        if (j == 10 or j == 11 or j == 12 or j == 14) and compSol_full == 'FB171':
                            continue
                        if (j == 15 or j == 17 or j == 19) and compSol_full == 'FB171':
                            continue
                        
                        if (j == 0  or j == 1 or j == 2 or j == 4) and compSol_full == 'PW113':
                            continue
                        if (j == 5 or j == 7 or j == 9) and compSol_full == 'PW113':
                            continue
                        if (j == 10 or j == 11 or j == 12 or j == 14) and compSol_full == 'PW113':
                            continue
                        if (j == 15 or j == 17 or j == 19) and compSol_full == 'PW113':
                            continue
                        
                        if (j == 0 or j == 1  or j == 2 or j == 4) and compSol_full == 'LS98':
                            continue
                        if (j == 5 or j == 7 or j == 9) and compSol_full == 'LS98':
                            continue
                        if (j == 10 or j == 11 or j == 12 or j == 14) and compSol_full == 'LS98':
                            continue
                        if (j == 15 or j == 17 or j == 19) and compSol_full == 'LS98':
                            continue
                        '''
                        curr_file = main_dir + 'S1_' + objs[j] + utilities[u] + '_' + compSol_full + '.csv'
                        df = pd.read_csv(curr_file, sep=',', skiprows=2, header=None).iloc[0, 1:]
                        s1_util[j, :] = pd.read_csv(curr_file, sep=',', skiprows=2, header=None).iloc[0, 1:]

                    s1_util_df = pd.DataFrame(s1_util, columns=hdrs)
                    out_filepath = main_dir + utilities_full[u] + '_' + compSol_full + '.csv'

                    s1_util_df.to_csv(out_filepath, sep=',', index=False)