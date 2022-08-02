# -*- coding: utf-8 -*-
"""
Created on Mon Apr  18 17:36:28 2022
Plots the CDFs for the three satisficing criteria of each of the three utilities.

@author: lbl59
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#import scipy.stats as stat

sns.set_style("darkgrid")

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

def calc_cdf(input_arr, N_SOLNS):
    """
    Performs the CDF calculation for an input array of performance objective values

    Parameters
    ----------
    input_arr : numpy array
        Sorted array of one performance objective across all perturbed instances.
    N_SOLNS : int
        Number of perturbed instances.

    Returns
    -------
    cdf : numpy array
        The cumulative distribution of the performance objective across all 
        perturbed instances.

    """
    # ref: https://www.statology.org/cdf-python/
    
    cdf = 1. * (np.arange(len(input_arr)) / (len(input_arr) - 1))
    return cdf

def plot_cdf(objs_by_rdm_dir, og_compSol, worst_robustness, 
             compSol, compSol_full, color_list):
    """
    Plots the CDF curves for each DU SOW across all perturbed instances.

    Parameters
    ----------
    objs_by_rdm_dir : string
        Directory where the raw DU Reevaluation output is stored.
    og_compSol : numpy matrix
        The matrix of the performance of the original compromise solution across
        all DU SOWs.
    worst_robustness : int
        The index of the perturbed instances with the worst robustness for a 
        utility/the region.
    compSol : string        
        Abbreviation of the compromise solution. LS98 for least-squares 
        (social planner) and PW113 for power index (pragmatist).
    compSol_full : string
        Title of the compromise solution.
    color_list : list
        List of line colors for the worst, original, and remaining plotlines.

    Returns
    -------
    None.

    """

    #objs_matrix = np.zeros((N_RDMS, N_SOLNS, 20), dtype='float')
    objs_matrix = np.zeros((N_SOLNS,20,N_RDMS), dtype='float')
    #cdf_matrix = np.zeros((N_RDMS, N_SOLNS, 20), dtype='float')
    cdf_matrix = np.zeros((N_SOLNS,20,N_RDMS), dtype='float')
    #sorted_matrix = np.zeros((N_RDMS, N_SOLNS, 20), dtype='float')
    sorted_matrix = np.zeros((N_SOLNS,20,N_RDMS), dtype='float')
    
    objs_matrix[1000,:,:] = og_compSol.T
    print(f"objs_matrix = {objs_matrix[1000,:,:]}")
    c_worst = color_list[0]
    c_og = color_list[1]
    c_others = color_list[2]
    
    for j in range(N_RDMS):
        filepathname = objs_by_rdm_dir + str(j) + '_sols0_to_' + str(N_SOLNS-1) + '.csv'
        objs_file = np.loadtxt(filepathname, delimiter=",")
        #objs_matrix[j,:N_SOLNS-1,:15] = objs_file
        objs_matrix[:N_SOLNS-1,:15,j] = objs_file
        
        #objs_file_wRegional = minimax(N_SOLNS-1, objs_matrix[j,:N_SOLNS-1,:])
        objs_file_wRegional = minimax(N_SOLNS-1, objs_matrix[:N_SOLNS-1,:,j])
        
        #objs_matrix[j,:N_SOLNS-1,:] = objs_file_wRegional
        objs_matrix[:N_SOLNS-1,:,j] = objs_file_wRegional

    for n in range(N_SOLNS):
        for n_objs in range(20):
            # sort the x-array before calculating its cumulative distribution
            if n_objs == 0 or n_objs == 5 or n_objs == 10 or n_objs == 15:
                # inverse the order of plotting for reliability since it is to be maximized
                sorted_matrix[n,n_objs,:] = np.sort(objs_matrix[n,n_objs,:]*(-100))
            else:
                sorted_matrix[n,n_objs,:] = np.sort(objs_matrix[n,n_objs,:]*100)
            
            cdf_matrix[n,n_objs,:] = calc_cdf(sorted_matrix[n,n_objs,:], N_RDMS)

    savefig_path = '/home/fs02/pmr82_0001/lbl59/Implementation_Uncertainty/post_processing_du/Figures/cdf_plots/'
    savefig_file = savefig_path + compSol + '_fullSOWs_CDF_indv.pdf'

    # Plot a 3x3 figure showing the CDFs for the three satisficing criteria for the 
    # three utilities
    fig, axes = plt.subplots(3,3,figsize=(12,12))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        for n_sol in range(N_SOLNS-1):
            x_data = sorted_matrix[n_sol, [0,1,4,5,6,9,10,11,14], :]
            y_data = cdf_matrix[n_sol, [0,1,4,5,6,9,10,11,14], :]

            ax.plot(x_data[i,:], y_data[i,:], c=c_others, linewidth=1.0, alpha=0.2)
            
    lab_w = 'Worst robustness'
    x_worst_W = sorted_matrix[worst_robustness[0], [0, 1, 4, 5, 6, 9, 10, 11, 14], :]
    y_worst_W = cdf_matrix[worst_robustness[0], [0, 1, 4, 5, 6, 9, 10, 11, 14], :]
    
    x_worst_D = sorted_matrix[worst_robustness[1], [0, 1, 4, 5, 6, 9, 10, 11, 14], :]
    y_worst_D = cdf_matrix[worst_robustness[1], [0, 1, 4, 5, 6, 9, 10, 11, 14], :]
    
    x_worst_F = sorted_matrix[worst_robustness[2], [0, 1, 4, 5, 6, 9, 10, 11, 14], :]
    y_worst_F = cdf_matrix[worst_robustness[2], [0, 1, 4, 5, 6, 9, 10, 11, 14], :]

    # original compsol
    x_og = sorted_matrix[1000, [0, 1, 4, 5, 6, 9, 10, 11, 14], :]
    y_og = cdf_matrix[1000, [0, 1, 4, 5, 6, 9, 10, 11, 14], :]
    
    for i, ax in enumerate(axes):
        ax.plot(x_og[i,:], y_og[i,:], c=c_og, linewidth=3, label=compSol_full)
        
        if i == 0 or i == 3:
            ax.set_ylabel('CDF', size=12)
            ax.set_xticks(np.linspace(-90, -100, num=5))
            ax.axvline(x=-98, ymin=0, ymax=1, color="black", linewidth=2, linestyle="--")
            #ax.invert_xaxis()
            ax.set_xlim(-100, -90)
            ax.set_xticklabels(['', '', '', '', ''])
            ax.set_yticks(np.linspace(0, 1, num=5))
            ax.set_yticklabels(['0', '', '', '', '1'])
            ax.plot(x_worst_W[i,:], y_worst_W[i,:], c=c_worst, linewidth=3, label=lab_w, linestyle="--")
            
        elif i == 1 or i == 4:
            ax.set_xticks(np.linspace(0, 80, num=8))
            ax.set_xticklabels(['', '', '', '', '', '', '', ''])
            ax.set_yticks(np.linspace(0, 1, num=5))
            ax.set_yticklabels(['', '', '', '', ''])
            ax.axvline(x=10, ymin=0, ymax=1, color="black", linewidth=2, linestyle="--")
            ax.set_xlim(0, 80)
            ax.plot(x_worst_D[i,:], y_worst_D[i,:], c=c_worst, linewidth=3, label=lab_w, linestyle="--")
            
        elif i == 2 or i == 5:
            ax.set_xticks(np.linspace(0, 80, num=8))
            ax.set_xticklabels(['', '', '', '', '', '', '', ''])
            ax.set_yticks(np.linspace(0, 1, num=5))
            ax.set_yticklabels(['', '', '', '', ''])
            ax.axvline(x=10, ymin=0, ymax=1, color="black", linewidth=2, linestyle="--")
            ax.set_xlim(0, 80)
            ax.plot(x_worst_F[i,:], y_worst_F[i,:], c=c_worst, linewidth=3, label=lab_w, linestyle="--")
            
        elif i == 6:
            ax.set_xlabel('Reliability', size=12)
            ax.set_xticks(np.linspace(-90, -100, num=5))
            ax.axvline(x=-98, ymin=0, ymax=1, color="black", linewidth=2, linestyle="--")
            ax.set_xlim(-100, -90)
            #ax.invert_xaxis()
            ax.set_xticklabels(['90%', '', '', '', '100%'])
            ax.set_ylabel('CDF', size=12)
            ax.set_yticks(np.linspace(0, 1, num=5))
            ax.set_yticklabels(['0', '', '', '', '1'])
            ax.plot(x_worst_W[i,:], y_worst_W[i,:], c=c_worst, linewidth=3, label=lab_w, linestyle="--")

        elif i == 7:
            ax.set_xlabel('Restriction freq', size=12)
            ax.axvline(x=10, ymin=0, ymax=1, color="black", linewidth=2, linestyle="--")
            ax.set_xticks(np.linspace(0, 80, num=8))
            ax.set_xticklabels(['0%', '', '', '', '', '', '', '80%'])
            ax.set_yticks(np.linspace(0, 1, num=5))
            ax.set_yticklabels(['', '', '', '', ''])
            ax.set_xlim(0, 80)
            ax.plot(x_worst_D[i,:], y_worst_D[i,:], c=c_worst, linewidth=3, label=lab_w, linestyle="--")

        elif i == 8:
            ax.set_xlabel('Worst-case cost', size=12)
            ax.axvline(x=10, ymin=0, ymax=1, color="black", linewidth=2, linestyle="--", 
                       label='Satisficing\ncriteria')
            ax.set_xticks(np.linspace(0, 80, num=8))
            ax.set_xticklabels(['0%', '', '', '', '', '', '', '80%'])
            ax.set_yticks(np.linspace(0, 1, num=5))
            ax.set_yticklabels(['', '', '', '', ''])
            ax.set_xlim(0, 80)
            ax.plot(x_worst_F[i,:], y_worst_F[i,:], c=c_worst, linewidth=3, label=lab_w, linestyle="--")
        '''
        else:
            ax.axvline(x=10, ymin=0, ymax=1, color="black", linewidth=2, linestyle="--")
            ax.set_yticks(np.linspace(0, 1, num=5))
            ax.set_yticklabels(['', '', '', '', ''])
        '''
    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    plt.savefig(savefig_file, bbox_inches="tight")

'''
Plotting starts here
'''
obj_names = ['REL_W', 'RF_W', 'INF_NPC_W', 'PFC_W', 'WCC_W', \
        'REL_D', 'RF_D', 'INF_NPC_D', 'PFC_D', 'WCC_D', \
        'REL_F', 'RF_F', 'INF_NPC_F', 'PFC_F', 'WCC_F', \
        'REL_R', 'RF_R', 'INF_NPC_R', 'PFC_R', 'WCC_R']

robustness_header = ['Watertown', 'Dryville', 'Fallsland', 'Regional']
compSol_colors = {'LS': ['indigo', 'indigo', 'thistle'],
                  'PW': ['darkgreen', 'darkgreen', 'palegreen']}

compSol_title = {'LS': 'Social planner',
                 'PW': 'Pragmatist'}

N_SOLNS = 1001
N_RDMS = 1000

compSols = ['PW113', 'LS98']
utils = ['_W', '_D', '_F', '_R']

raw_output_dir = '/home/fs02/pmr82_0001/lbl59/Implementation_Uncertainty/WaterPaths_duReeval/'
main_dir = '/home/fs02/pmr82_0001/lbl59/Implementation_Uncertainty/post_processing_du/DU_reeval_output_Apr2022/'

# make changes here
compSol = 'PW'
compSol_full = 'PW113'

# perturbed compSol objectives
filepath_objs_pt = raw_output_dir + 'Objectives_' + compSol + '_perturbed_Apr2022/Objectives_RDM'

# original compSol objectives
filepath_objs_og = main_dir + compSol_full + '/original_compromise_acrossSoln_' + compSol_full + '.csv'

og_compSol = np.loadtxt(filepath_objs_og, delimiter=",")
worst_robustness = main_dir + compSol_full + '/worst_robustness_idx_' + \
    compSol_full + '.csv'

idx_worst_df = pd.read_csv(worst_robustness, sep=",", header=0)
idx_worst = idx_worst_df.iloc[:,1]


plot_cdf(filepath_objs_pt, og_compSol, idx_worst, compSol_full,
         compSol_title[compSol], compSol_colors[compSol])

