# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 12:47:51 2020

Plots the pathways for the original SP compromise and its pathways when Fallsland
experiences its individual worst robustness.

Does the plotting under two DU scenarios:
        - exp: 'Expected' evaporation and demand growth rates
        - challenge: 'Challenging' evaporation and demand growth rates

Plotting for Watertown and Dryville occurs separately due to differing number of clusters

@author: dgold
@edited: lbl59
"""

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib


def cluster_pathways(solution, compSol_full, mode, rdm, utility, num_clusters):
    """
    Clusters infrastructure pathways be the week each option is constructed
    creates "representative pathways" for diagnostics and communication

    Parameters:
        solution: name of the solution to be plotted (should be folder name)
        compSol_full: name of the compromise solution to plot
        mode: baseline, worst regional robustness or best regional robustness
        utility: a string (all lowercase) of the name of the utility of interest
        num_clusters: number of clusters used (should be carefully assessed)

    returns:
        cluster_pathways: a 3-d list containing the pathways for each cluster
        cluster_medians: an array containing median construction weeks for each
        inf option in each cluster

    """
    filepath ='/home/fs02/pmr82_0001/lbl59/Implementation_Uncertainty/WaterPaths_duReeval/' + \
        mode + '_' + compSol_full + '_' + solution + '/'
    fileloc = filepath + 'Pathways/Pathways_s' + solution + '_RDM' + str(rdm) + '.out'
    
    pathways_df = pd.read_csv(fileloc, sep='\t')

    # reformat for clustering (need an array with each row a realization and
    # each column a different infrastructure option. elements are const weeks)
    cluster_input = np.ones([500,13])*2344

    # loop through each realization
    for real in range(0,500):
    # extract the realization
        current_real = pathways_df[pathways_df['Realization']==real]
        # find the infrastructure option (ids 0-2 are already built, 6 is off)
        for inf in [3,4,5,7,8,9,10,11,12]:
            if not current_real.empty:
                for index, row in current_real.iterrows():
                    if row['infra.']==inf:
                        cluster_input[real, inf] = row['week']

    # post process to remove inf options never constructed and normalize weeks
    # to [0-1] by dividing by total weeks, 2344
    cluster_input = cluster_input[:,[4,5,7,8,9,10,11,12]]/2344
    #inf_options = inf_options[4:6]+ inf_options[7:13]

    # extract columns for each utility
    if utility == 'watertown':
        # watertown has NRR, CRR_low, CRR_high, WR, WRII
        cluster_input = cluster_input[:,[0, 2, 3, 4, 5]]
    elif utility == 'dryville':
        # dryville has SCR, DR
        cluster_input = cluster_input[:,[1, 6]]
    else:
        # fallsland has NRR, FR
        cluster_input = cluster_input[:,[0, 7]]

    # k-means clustering
    k_means = KMeans(init='k-means++', n_clusters = num_clusters, n_init=10)

    k_means.fit(cluster_input)
    k_means_cluster_centers = k_means.cluster_centers_
    k_means_labels = pairwise_distances_argmin(cluster_input, k_means_cluster_centers)

    # assign each realization to a pathway, and calculate the median week
    # each infrstructure option is constructed in each cluster
    cluster_pathways = []
    cluster_medians = []
    for i in range(0, num_clusters):
        current_cluster =  cluster_input[k_means_labels==i,:]*2344
        cluster_pathways.append(current_cluster)
        current_medians = np.zeros(len(current_cluster[0,:]))
        
        for j in range(0, len(current_cluster[0,:])):
            current_medians[j]= np.median(current_cluster[:,j])
        
        cluster_medians.append(current_medians)
    # sort clusters by average of medians to get heavy, mod and light clusters
    
    cluster_means = [np.mean(cluster_medians[0]), np.mean(cluster_medians[1])]

    sorted_indicies = np.argsort(cluster_means)
    
    cluster_medians = np.vstack((cluster_medians[sorted_indicies[1]],
                                     cluster_medians[sorted_indicies[0]]))
    
    return cluster_pathways, cluster_medians

def plot_single_pathway(cluster_medians, cluster_pathways, inf_options_idx,
                        c, cmap, ax, y_offset, plot_legend):
    """
    Makes a plot of an infrastructure Pathway

    Parameters:
        cluster_medias: an array with median weeks each option is built
        cluster_pathways: an array with every pathway in the cluster
        inf_options: an array with numbers representing each option (y-axis vals)
        should start at zero to represent the "baseline"
        c: color to plot pathway
        ax: axes object to plot pathway
        inf_names: a list of strings with the names of each pathway

    """
    # get array of the infrastructure options without baseline
    inf_options_idx_no_baseline=inf_options_idx[1:]

    sorted_inf = np.argsort(cluster_medians)

    # plot heatmap of construction times
    cluster_pathways = np.rint(cluster_pathways/45)
    inf_im = np.zeros((45, np.shape(cluster_pathways)[1]+1))

    for k in range(1,np.shape(cluster_pathways)[1]+1) :
        for i in range(0,45):
            for j in range(0, len(cluster_pathways[:,k-1])):
                if cluster_pathways[j,k-1] == i:
                    inf_im[i,k] +=1
        print(min(inf_im[:,k]))
        print(max(inf_im[:,k]))

    ax.imshow((inf_im.T)/0.85, cmap=cmap, aspect='auto', alpha = 0.6)
    
    # sort by construction order
    #cluster_medians = np.rint(cluster_medians/45)
    #sorted_inf = np.argsort(cluster_medians)

    # plot pathways
    # create arrays to plot the pathway lines. To ensure pathways have corners
    # we need an array to have length 2*num_inf_options
    pathway_x = np.zeros(len(cluster_medians)*2+2)
    pathway_y = np.zeros(len(cluster_medians)*2+2)

    # to make corners, each inf option must be in the slot it is triggered, and
    # the one after
    cluster_medians = np.rint(cluster_medians/45)
    for i in range(0,len(cluster_medians)):
        for j in [1,2]:
            pathway_x[(i*2)+j] = cluster_medians[sorted_inf[i]]
            pathway_y[(i*2)+j+1] = inf_options_idx_no_baseline[sorted_inf[i]]

    # end case
    pathway_x[-1] = 45

    # plot the pathway line
    ax.plot(pathway_x, pathway_y+y_offset, color=c, linewidth=5,
            alpha = .9, zorder=1)

    ax.set_xlim([0,44])


def create_cluster_plots(w_meds, d_meds, f_meds, w_pathways, d_pathways,
                         f_pathways, n_clusters, cluster_colors, cmaps, fig,
                         gspec, fig_col, plot_legend):
    """
    creates a figure with three subplots, each representing a utility

    Parameters:
        w_meds: median values for each cluster for watertown
        d_meds: median values for each cluster for dryville
        f_meds: median values for each cluster for fallsland
        w_pathways: all pathways in each watertown cluster
        d_pathways: all pathways in each dryville cluster
        f_pathways: all pathways in each fallsland cluster
        n_clusters: number of clusters
        cluster_colors: an array of colors for each cluster
        cmaps: an array of colormaps for coloring the heatmap
        fig: a figure object for plotting
        fig_dims: an array with the number of rows and columns of subplots

        NOTE: DOES NOT SAVE THE FIGURE
    """

    watertown_inf = ['Baseline', 'New River\nReservoir',
                     'College Rock\nExpansion Low',
                     'College Rock\nExpansion High', 'Water Reuse',
                     'Water Reuse II']
    dryville_inf = ['', 'Baseline', '', 'Sugar Creek\nReservoir', '', 'Water Reuse']
    fallsland_inf = ['', 'Baseline', '', 'New River\nReservoir', '', 'Water Reuse']
    
    #fig.text(0.5, 0.01, 'Years', ha='center', va='center')
    #fig.text(0.01, 0.5, 'Infrastructure options number', ha='center', va='center', rotation='vertical')

    ax1 = fig.add_subplot(gspec[0, fig_col])
    ax2 = fig.add_subplot(gspec[1, fig_col])
    ax3 = fig.add_subplot(gspec[2, fig_col])

    y_offsets = [-0.15, 0, 0.15]

    '''
    w_plot_order = np.argsort([np.mean(w_meds[0]), np.mean(w_meds[1]),
                               np.mean(w_meds[2])])
    d_plot_order = np.argsort([np.mean(d_meds[0]), np.mean(d_meds[1]),
                               np.mean(d_meds[2])])
    f_plot_order = np.argsort([np.mean(f_meds[0]), np.mean(f_meds[1]),
                              np.mean(f_meds[2])])
    '''

    for i in np.arange(n_clusters):
        
        plot_single_pathway(w_meds[i], w_pathways[i], np.array([0,1,2,3,4,5]),
                              cluster_colors[i], cmaps[i], ax1, 
                              y_offsets[i], plot_legend)
        
        plot_single_pathway(d_meds[i], d_pathways[i], np.array([0,1,2]),
                              cluster_colors[i], cmaps[i], ax2, 
                              y_offsets[i], plot_legend)
        
        plot_single_pathway(f_meds[i], f_pathways[i], np.array([0,1,2]),
                              cluster_colors[i], cmaps[i], ax3, 
                              y_offsets[i],  plot_legend)
        
        if fig_col == 0:
            ax1.set_ylabel('Watertown', fontsize=14)
            ax1.set_yticks(np.arange(0, 6))
            ax1.set_yticklabels(watertown_inf)
            
            ax2.set_ylabel('Dryville', fontsize=14)
            ax2.set_yticks(np.linspace(-0.65, 2.15, 6))
            ax2.set_yticklabels(dryville_inf)
            
            #ax3.set_xlabel('Challenging\nDU scenario', fontsize=14)
            ax3.set_xlabel('Worst Fallsland\nrobustness', fontsize=14)
            ax3.set_ylabel('Fallsland', fontsize=14)
            ax3.set_yticks(np.linspace(-0.65, 2.15, 6))
            ax3.set_yticklabels(fallsland_inf)
            
        else:
            ax1.set_yticks(np.arange(0, 6))
            ax1.set_yticklabels(['', '', '', '', '', ''])
            
            ax2.set_yticks(np.linspace(-0.65, 2.15, 6))
            ax2.set_yticklabels(['', '', '', '', '', ''])
            
            ax3.set_yticks(np.linspace(-0.65, 2.15, 6))
            ax3.set_yticklabels(['', '', '', '', '', ''])
            
    '''
    if plot_legend:
        ax1.legend(['Light inf.', 'High inf.'],
                   loc='upper left')
    '''
    
    ax1.tick_params(axis = "y", which = "both", left = False, right = False)
    ax2.tick_params(axis = "y", which = "both", left = False, right = False)
    ax3.tick_params(axis = "y", which = "both", left = False, right = False)

    #if fig_row == 1:
    #    ax1.set_title('Watertown Pathways')
    #    ax2.set_title('Dryville Pathways')
    #    ax3.set_title('Fallsland Pathways')
    '''
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)
    if fig_col == 3:
        ax1.set_axis_off()
        cbar1 = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmaps[0]), ax=[ax1,ax2,ax3], 
                             shrink=0.89, fraction=0.1, pad=0.10, ticks=[0.2,0.4,0.6,0.8,1.0])
        cbar1.set_label(r'Constr. freq. $\rightarrow$', labelpad=12, fontsize=12)
    '''  
    plt.tight_layout()

compSol = 'LS'
compSol_full = 'LS98'
compSol_num = 2   # FB is 0, PW is 1, LS is 2

worst_robustness_W = [610, 260, 91]
worst_robustness_D = [543, 250, 433]
worst_robustness_F = [143, 337, 81]
worst_robustness_R = [543, 250, 433]

bad_scenario = 223   # evap multiplier = 1.2, demand multiplier = 1.95
baseline_scenario = 229     # evap multiplier = 1.0, demand multiplier = 0.99

soln_idxs = ['0', str(worst_robustness_W[compSol_num]), str(worst_robustness_D[compSol_num]), 
               str(worst_robustness_F[compSol_num])]
soln_modes = ['original', 'Watertown', 'Dryville', 'Fallsland']

rdm_modes = ['exp', 'challenge']
rdm_idx = [baseline_scenario, bad_scenario]

path_colors = {'FB': ['darkgoldenrod', 'sandybrown', 'navajowhite'], \
               'LS': ['indigo', 'mediumslateblue', 'thistle'], \
               'PW': ['darkgreen', 'mediumseagreen', 'palegreen']}

#rdms = [bad_scenarios[compSol_num], baseline_scenarios[compSol_num], optimistic_scenarios[compSol_num]]
rdms = [bad_scenario, baseline_scenario]

# for r in range(len(rdms)):
for i in range(len(rdm_idx)):
    rdm = rdm_idx[i]
    rdm_mode = rdm_modes[i] 
    #rdm = rdms[r]
    
    if i == 0:
        title_label = 'Expected scenario'    
    elif i == 1:
        title_label = 'Challenging scenario'

    
    fig_filepath = '/home/fs02/pmr82_0001/lbl59/Implementation_Uncertainty/post_processing_du/Figures/pathways/'
    fig_fileloc = fig_filepath + compSol_full + title_label + '_pathways_indvF_cbar.pdf'                                                                                                                                     
    
    fig = plt.figure(figsize=(12,10), dpi=300)
    gspec = fig.add_gridspec(nrows=3, ncols=4, height_ratios =[1,1,1])
    
   
    solution = soln_idxs[3]    #'Drought + High demand scenario'
    mode = soln_modes[3]   
    
    heavy_inf_color = path_colors[compSol][0]
    #mid_inf_color = path_colors[compSol][1]
    light_inf_color = path_colors[compSol][2]

    num_clust = 2
    
    watertown_cluster_pathways, watertown_cluster_meds = cluster_pathways(solution, compSol_full, rdm_mode, rdm, 'watertown', num_clust)
    dryville_cluster_pathways, dryville_cluster_meds = cluster_pathways(solution, compSol_full, rdm_mode, rdm, 'dryville', num_clust)
    fallsland_cluster_pathways, fallsland_cluster_meds = cluster_pathways(solution, compSol_full, rdm_mode, rdm, 'fallsland', num_clust)
    print(f"solution = {solution} done")
    
    create_cluster_plots(watertown_cluster_meds, dryville_cluster_meds,
                         fallsland_cluster_meds, watertown_cluster_pathways,
                         dryville_cluster_pathways, fallsland_cluster_pathways,
                          num_clust, [light_inf_color, heavy_inf_color],
                         ['bone_r', 'bone_r'], fig, gspec, 3, True)

    
    plt.suptitle(title_label, fontsize=16, y=1.00)
    plt.tight_layout()
    plt.savefig(fig_fileloc)
