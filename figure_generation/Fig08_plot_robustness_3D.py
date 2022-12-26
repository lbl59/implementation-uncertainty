# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 17:36:28 2022

Plots the robustness tradeoff between different utilities and identifies the
worst robustness fo each utility and the region as a whole.

@author: lbl59
"""
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import seaborn as sns
sns.set_theme(style="whitegrid")

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

'''
Begin changing values here
'''

compSol_LS = 'LS'
compSol_full_LS = 'LS98'
compSol_num_LS = 0

compSol_PW = 'PW'
compSol_full_PW = 'PW113'
compSol_num_PW = 1

compSol_labels = ['Social planner', 'Pragmatist']
compSol_og_colors = ['plum', 'mediumseagreen']
compSol_worst_colors = ['purple', 'darkgreen']

c_worst_cols = ['gold', 'thistle', 'palegreen']

worst_robustness_W = [91, 260]
worst_robustness_D = [433, 250]
worst_robustness_F = [81, 337]
worst_robustness_R = [433, 250]

W_shape = 'v'
D_shape = 'd'
F_shape = 4
R_shape = 'X'

compSol_label_LS = compSol_labels[compSol_num_LS]
compSol_og_LS  = compSol_og_colors[compSol_num_LS]
compSol_worst_LS  = compSol_worst_colors[compSol_num_LS]

compSol_label_PW = compSol_labels[compSol_num_PW]
compSol_og_PW  = compSol_og_colors[compSol_num_PW]
compSol_worst_PW  = compSol_worst_colors[compSol_num_PW]

worst_idx_W_LS = worst_robustness_W[compSol_num_LS]
worst_idx_D_LS = worst_robustness_D[compSol_num_LS]
worst_idx_F_LS = worst_robustness_F[compSol_num_LS]
worst_idx_R_LS = worst_robustness_R[compSol_num_LS]

worst_idx_W_PW = worst_robustness_W[compSol_num_PW]
worst_idx_D_PW = worst_robustness_D[compSol_num_PW]
worst_idx_F_PW = worst_robustness_F[compSol_num_PW]
worst_idx_R_PW = worst_robustness_R[compSol_num_PW]

utilities = ['Watertown', 'Dryville', 'Fallsland', 'Regional']
utils_indv = ['Watertown', 'Dryville', 'Fallsland']

x_lab = 'Watertown robustness (%)\n$\longrightarrow$'
x_key = 'Watertown'
y_lab = 'Dryville robustness (%)\n$\longrightarrow$'
y_key = 'Dryville'
z_lab = 'Fallsland robustness (%)\n$\longrightarrow$'
z_key = 'Fallsland'

'''
Get robustness data
'''
directory_LS = '/home/fs02/pmr82_0001/lbl59/Implementation_Uncertainty/'+\
    'post_processing_du/DU_reeval_output_Apr2022/' + compSol_full_LS + '/'
directory_PW = '/home/fs02/pmr82_0001/lbl59/Implementation_Uncertainty/'+\
    'post_processing_du/DU_reeval_output_Apr2022/' + compSol_full_PW + '/'
    
# change depending on compromise solution being analyzed
filename_robustness_LS = 'robustness_perturbed_og_' + compSol_full_LS + '.csv'
robustness_df_LS = pd.read_csv(directory_LS + filename_robustness_LS, sep=',', names=utilities)

filename_robustness_PW = 'robustness_perturbed_og_' + compSol_full_PW + '.csv'
robustness_df_PW = pd.read_csv(directory_PW + filename_robustness_PW, sep=',', names=utilities)

'''
Get robustness values data
'''
x_data_LS = (robustness_df_LS[x_key].values)*100
y_data_LS = (robustness_df_LS[y_key].values)*100
z_data_LS = (robustness_df_LS[z_key].values)*100

x_data_PW = (robustness_df_PW[x_key].values)*100
y_data_PW = (robustness_df_PW[y_key].values)*100
z_data_PW = (robustness_df_PW[z_key].values)*100

fig = plt.figure(figsize=(8,8))

ax = fig.add_subplot(projection='3d')

all_colors = 'lightgrey'
color_worst_LS = compSol_worst_colors[compSol_num_LS]
color_worst_PW = compSol_worst_colors[compSol_num_PW]

color_og_LS = compSol_og_colors[compSol_num_LS]
color_og_PW = compSol_og_colors[compSol_num_PW]

label_W_og = 'Original Watertown robustness'
label_D_og = 'Original Dryville robustness'
label_F_og = 'Original Fallsland robustness'
label_R_og = 'Original regional robustness'

label_W_w = 'Worst Watertown robustness'
label_D_w = 'Worst Dryville robustness'
label_F_w = 'Worst Fallsland robustness'
label_R_w = 'Worst regional robustness'

p = ax.scatter3D(x_data_LS, y_data_LS, z_data_LS, s=90, c=all_colors, alpha=0.2)
ax.scatter3D(x_data_PW, y_data_PW, z_data_PW, s=90, c=all_colors, alpha=0.2)

ax.scatter3D(x_data_LS[worst_idx_W_LS], y_data_LS[worst_idx_D_LS], z_data_LS[worst_idx_F_LS], 
             s=280, marker='X',
             c=color_worst_LS, alpha=1.0, edgecolors=color_worst_LS, linewidths=2,
             label='Worst SP robustness')
'''
ax.scatter3D(x_data_LS[worst_idx_D_LS], y_data_LS[worst_idx_D_LS], z_data_LS[worst_idx_D_LS], 
             s=180, marker=D_shape,
             c=color_worst_LS, alpha=0.8, edgecolors=color_worst_LS, linewidths=2,
             label=label_D_w)

ax.scatter3D(x_data_LS[worst_idx_F_LS], y_data_LS[worst_idx_F_LS], z_data_LS[worst_idx_F_LS], 
             s=180, marker=F_shape,
             c=color_worst_LS, alpha=0.8, edgecolors=color_worst_LS, linewidths=2,
             label=label_F_w)

ax.scatter3D(x_data_LS[worst_idx_R_LS], y_data_LS[worst_idx_R_LS], z_data_LS[worst_idx_R_LS], 
             s=180, marker=R_shape,
             c=color_worst_LS, alpha=0.8, edgecolors=color_worst_LS, linewidths=2,
             label=label_R_w)
'''
ax.scatter3D(x_data_PW[worst_idx_W_PW], y_data_PW[worst_idx_D_PW], z_data_PW[worst_idx_F_PW], 
             s=280, marker='X',
             c=color_worst_PW, alpha=1.0, edgecolors=color_worst_PW, linewidths=2,
             label=label_W_w)
'''
ax.scatter3D(x_data_PW[worst_idx_D_PW], y_data_PW[worst_idx_D_PW], z_data_PW[worst_idx_D_PW], 
             s=180, marker=D_shape,
             c=color_worst_PW, alpha=0.8, edgecolors=color_worst_PW, linewidths=2,
             label=label_D_w)

ax.scatter3D(x_data_PW[worst_idx_F_PW], y_data_PW[worst_idx_F_PW], z_data_PW[worst_idx_F_PW], 
             s=180, marker=F_shape,
             c=color_worst_PW, alpha=0.8, edgecolors=color_worst_PW, linewidths=2,
             label=label_F_w)

ax.scatter3D(x_data_PW[worst_idx_R_PW], y_data_PW[worst_idx_R_PW], z_data_PW[worst_idx_R_PW], 
             s=180, marker=R_shape,
             c=color_worst_PW, alpha=0.8, edgecolors=color_worst_PW, linewidths=2,
             label=label_R_w)
'''
ax.scatter3D(x_data_LS[1000], y_data_LS[1000], z_data_LS[1000], 
             s=280, marker='^',
             c=color_worst_LS, alpha=1.0, edgecolors=color_worst_LS, linewidths=2,
             label='Social planner')

ax.scatter3D(x_data_PW[1000], y_data_PW[1000], z_data_PW[1000], 
             s=280, marker='^',
             c=color_worst_PW, alpha=1.0, edgecolors=color_worst_PW, linewidths=2,
             label='Pragmatist')

ax.scatter3D(100, 100, 100, 
             s=260, marker='*',linewidths=4,
             c='k', alpha=1.0, label='Ideal point')

ax.set_xlabel(x_lab, labelpad=10)

ax.set_ylabel(y_lab, labelpad=10)

ax.set_zlabel(z_lab, labelpad=10)
#ax.set_zticks(np.linspace(97, 100, 6))
#ax.set_zticklabels(['','97', '98', '99', '100',''])

# change depending on objective you want to plot
title = 'Robustness across all perturbed instances\nfor all utilities'

'''
robustness_savefile = '/home/fs02/pmr82_0001/lbl59/Implementation_Uncertainty/' + \
    'post_processing_du/Figures/3d_plots/robustness3d_' + compSol_full + '_' + compSol[compSol_num] + '.pdf'
'''
robustness_savefile = '/home/fs02/pmr82_0001/lbl59/Implementation_Uncertainty/' + \
    'post_processing_du/Figures/3d_plots/robustness3d_allComp.pdf'
    
plt.legend(bbox_to_anchor=(0.3, 1.0), loc='upper right', prop={'size': 9}, markerscale=0.9, ncol=1)
plt.title(title, size=12)
plt.savefig(robustness_savefile)
plt.show()
