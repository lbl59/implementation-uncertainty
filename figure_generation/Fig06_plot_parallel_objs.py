# plot deviated regional objectives with respect to original dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns
import matplotlib
from matplotlib import cm

sns.set_theme(style="whitegrid")

def set_ticks_for_axis(dim, ax_i, ticks):
    min_val, max_val, v_range = min_max_range[obj_labels[dim]]

    if dim != 2:
        min_val = min_val * 100
        max_val = max_val * 100
        v_range = v_range * 100

    step = ((v_range))/float(ticks-1)

    tick_labels = np.array([round(min_val + step*i, 1) for i in range(ticks)])

    if dim == 0:
        tick_labels = np.flip(tick_labels)

    tick_labels = tick_labels.astype(int)
    tick_labels_woot = []
    for i in range(len(tick_labels)):
        if i == 2:
            tick_labels_woot.append(str(tick_labels[i]))
        elif i != 2:
            tick_labels_woot.append(str(tick_labels[i]) + '%')

    norm_min = 0.0
    norm_range = 1.0
    norm_step = norm_range/float(ticks-1)
    ticks = [round(norm_min + norm_step*i, 1) for i in range(ticks)]
    ax_i.yaxis.set_ticks(ticks)
    ax_i.set_yticklabels(tick_labels_woot, fontsize=8)
    ax_i.tick_params(axis='y', left=False, right=False)
    #ax_i.set_ylim([norm_min, max(refset_norm[:,dim])])

utilities = ['Watertown', 'Fallsland', 'Dryville', 'regional']
dv_names = ['RT_W', 'RT_D', 'RT_F', 'TT_D', 'TT_F', 'LMA_W', 'LMA_D', 'LMA_F',\
            'RC_W', 'RC_D', 'RC_F', 'IT_W', 'IT_D', 'IT_F', 'IP_W', 'IP_D', \
            'IP_F', 'INF_W', 'INF_D', 'INF_F']

obj_names = ['REL_W', 'RF_W', 'INF_NPC_W', 'PFC_W', 'WCC_W', \
        'REL_D', 'RF_D', 'INF_NPC_D', 'PFC_D', 'WCC_D', \
        'REL_F', 'RF_F', 'INF_NPC_F', 'PFC_F', 'WCC_F', \
        'REL_R', 'RF_R', 'INF_NPC_R', 'PFC_R', 'WCC_R']

# modify these depending on compsol being analyzed
compSol = 'PW'
compSol_full = 'PW113'
compSol_num = 1

colors_worst = ['saddlebrown', 'darkgreen', 'darkviolet']
colors_back = ['mocassin', 'palegreen', 'thistle']
colormaps = ['YlOrBr', 'Greens', 'Purples']
comp = ['Fallback bargaining', 'Power index', 'Least squares']

worst_robustness_W = [610, 260, 91]
worst_robustness_D = [543, 250, 433]
worst_robustness_F = [143, 337, 81]
worst_robustness_R = [543, 250, 433]

reg_idx_worst_W = worst_robustness_W[compSol_num]
reg_idx_worst_D = worst_robustness_D[compSol_num]
reg_idx_worst_F = worst_robustness_F[compSol_num]
reg_idx_worst_R = worst_robustness_R[compSol_num]

# get obj and dv filepaths then load the data
dvs_directory = '/home/fs02/pmr82_0001/lbl59/Implementation_Uncertainty/WaterPaths_duReeval/IU_Samples/'
dvs_filepath_p = dvs_directory + 'IU_allMeasures_' + compSol + '.csv'
dvs_filepath_o = dvs_directory + compSol + '_soln.csv'

objs_directory = '/home/fs02/pmr82_0001/lbl59/Implementation_Uncertainty/post_processing_du/DU_reeval_output_Apr2022/' \
    + compSol_full + '/'
objs_filepath_p = objs_directory + 'meanObjs_acrossRDM_' + compSol_full + '.csv'
objs_filepath_o = objs_directory + 'original_compromise_acrossRDM_' + compSol_full + '.csv'

df_dvs_p = pd.read_csv(dvs_filepath_p, index_col=False, names = dv_names)
df_dvs_o = pd.read_csv(dvs_filepath_o, index_col=False, names = dv_names)
df_dvs = pd.concat([df_dvs_p, df_dvs_o], ignore_index=True, axis=0)

df_objs_p = pd.read_csv(objs_filepath_p, index_col=False, names = obj_names)
df_objs_o = pd.read_csv(objs_filepath_o, index_col=False, names = obj_names)
df_objs = pd.concat([df_objs_p, df_objs_o], ignore_index=True, axis=0)

# objective value subsets
objs_W = ['REL_W', 'RF_W', 'INF_NPC_W', 'PFC_W', 'WCC_W']
objs_D = ['REL_D', 'RF_D', 'INF_NPC_D', 'PFC_D', 'WCC_D']
objs_F = ['REL_F', 'RF_F', 'INF_NPC_F', 'PFC_F', 'WCC_F']
objs_R = ['REL_R', 'RF_R', 'INF_NPC_R', 'PFC_R', 'WCC_R']

objs_subset_dict = {'W': objs_W, 'D': objs_D, 'F': objs_F, 'R': objs_R}
util_list = ['Watertown', 'Dryville', 'Fallsland', 'Regional']
util_short_list = ['W', 'D', 'F', 'R']


# load robustness data
filename_robustness = objs_directory + '/robustness_perturbed_og_' + compSol_full + '.csv'
robustness_data = pd.read_csv(filename_robustness, sep=',', names=utilities, index_col=False)
regional_data = robustness_data[utilities].min(axis=1)
robustness_data['regional_scaled'] = regional_data.values

min_objs = [0.96, 0.0, 0.0, 0.0, 0.0, 0.0]

max_objs = [1.0, 0.3, 100.0, 1.0, 1.0]

for i in range(len(util_short_list)):
    util_name = util_list[i]
    util_short = util_short_list[i]
    objs_subset = objs_subset_dict[util_short]
    reg_idx_worst = 0
    if util_short == 'W':
        reg_idx_worst = reg_idx_worst_W
    elif util_short == 'D':
        reg_idx_worst = reg_idx_worst_D
    elif util_short == 'F':
        reg_idx_worst = reg_idx_worst_F
    elif util_short == 'R':
        reg_idx_worst = reg_idx_worst_R

    parallel_plot_title = util_list[i]
    # sort objectives by regional robustness
    refset_df = df_objs[objs_subset]
    refset_df['robustness_scaled'] = (regional_data)/0.85
    print(min(refset_df['robustness_scaled']))
    print(max(refset_df['robustness_scaled']))
    refset_df = refset_df.sort_values('robustness_scaled')

    reg_plot = refset_df[['robustness_scaled']].values

    refset = refset_df.to_numpy()
    nrow = refset.shape[0]

    # process and normalize the reference set
    min_max_range = {}

    refset_norm = np.copy(refset[:, 0:5])
    # identify and list the objectives of the reference set
    obj_labels = ['Reliability', 'Restriction\nfrequency', 'Infrastructure net\npresent cost ($mil)',
                  'Peak financial\ncost', 'Worst-case\ncost']

    for i in range(len(obj_labels)):
        if i == 0:
            refset_norm[:,i] = np.true_divide(max_objs[i] - refset_norm[:,i], (max_objs[i] - min_objs[i]))
        #elif i == 2:
            #refset_norm[:,i] = np.zeros(refset_norm.shape[0])
        elif i == 2 and compSol == 'PW':
            refset_norm[:,i] = [0]*len(refset_norm[:,i])
        else:
            refset_norm[:,i] = np.true_divide(refset_norm[:,i] - min_objs[i], (max_objs[i] - min_objs[i]))

        min_max_range[obj_labels[i]] = [min_objs[i], max_objs[i], (max_objs[i] - min_objs[i])]


    # create an array of integers ranging from 0 to the number of objectives
    x = [i for i, _ in enumerate(obj_labels)]

    # sharey=False indicates that all the subplot y-axes will be set to different values
    fig, ax  = plt.subplots(1,len(x)-1, sharey=False, figsize=(10,3))
    cmap = matplotlib.cm.get_cmap(colormaps[compSol_num])

    for i, ax_i in enumerate(ax):
        for d in range(refset_norm.shape[0]):
                ax_i.plot(obj_labels, refset_norm[d, :], color=cmap(reg_plot[d][0]), alpha=0.6, linewidth=5)
                
                ax_i.grid(False)
        ax_i.set_xlim([x[i], x[i+1]])
        ax_i.tick_params(axis='x', which='major', pad=0.2)

    for i, ax_i in enumerate(ax):
        ax_i.plot(obj_labels, refset_norm[1000, :], color=colors_back[compSol_num], alpha=1.0, linewidth=3,
                  label=comp[compSol_num])   # change here!
        ax_i.grid(False)
        ax_i.set_xlim([x[i], x[i+1]])
        ax_i.tick_params(axis='x', which='major', pad=0.2)

    for i, ax_i in enumerate(ax):
        ax_i.plot(obj_labels, refset_norm[reg_idx_worst, :], color=colors_back[compSol_num], alpha=1.0, linewidth=3,
                  linestyle='--', label="Worst robustness")
        ax_i.grid(False)
        ax_i.set_xlim([x[i], x[i+1]])
        ax_i.tick_params(axis='x', which='major', pad=0.2)
        ax_i.spines['top'].set_visible(False)
        ax_i.spines['bottom'].set_visible(False)
        ax_i.tick_params(axis='x', labelsize=8)

    for dim, ax_i in enumerate(ax):
        ax_i.xaxis.set_major_locator(ticker.FixedLocator([dim]))
        set_ticks_for_axis(dim, ax_i, ticks=2)

    ax2 = plt.twinx(ax[-1])
    dim = len(ax)
    ax2.xaxis.set_major_locator(ticker.FixedLocator([x[-2], x[-1]]))
    set_ticks_for_axis(dim, ax2, ticks=2)
    ax2.spines['top'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.grid(False)
    ax2.patch.set_alpha(0.01)
    ax2.set_xticklabels([obj_labels[-2], obj_labels[-1]], fontsize=6)
    ax2.tick_params(axis='x', which='major', pad=0.2)
    ax2.tick_params(axis='y', left=False, right=False)
    ax[3].tick_params(axis='y', left=False, right=False)
    plt.subplots_adjust(wspace=0, hspace=0.2, left=0.1, right=0.85, bottom=0.1, top=0.9)
    ax[3].legend(bbox_to_anchor=(1.0, 1.0), loc='upper right', prop={'size': 10})
    ax[0].set_ylabel("$\leftarrow$ Direction of preference", fontsize=8)
    
    norm = matplotlib.colors.Normalize(vmin=0.2, vmax=1.0)
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax[3], fraction=0.03, pad=0.05, ticks=[0.2,0.4,0.6,0.8,1.0])
    cbar.set_label(r'Robustness (%) $\rightarrow$', labelpad=12, fontsize=12)
    plt.suptitle(parallel_plot_title, fontsize=10, y=1.001)
    fig_filename = 'Figures/parallel_plots/objs_reeval_' + util_short + '_' + compSol_full + '_indv_cbar.pdf'

    plt.savefig(fig_filename)
    plt.show()
