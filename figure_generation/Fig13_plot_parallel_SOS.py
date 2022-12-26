# plot deviated regional objectives with respect to original dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns

sns.set_theme(style="whitegrid")

def set_ticks_for_axis(dim, ax_i, ticks):
    min_val, max_val, v_range = min_max_range[dv_labels[dim]]

    step = ((v_range))/float(ticks-1)

    tick_labels = np.array([round(min_val + step*i, 1) for i in range(ticks)])

    tick_labels = tick_labels.astype(int)
    
    tick_labels_woot = []
    for i in range(len(tick_labels)):
        tick_labels_woot.append(str(tick_labels[i]) + '%')
    
    if dv_labels[dim] == 'Restriction\ntrigger' or dv_labels[dim] == 'Insurance\ntrigger' or dv_labels[dim] == 'Infrastructure\ntrigger' or dv_labels[dim] == 'Transfer\ntrigger':
        tick_labels_woot = np.flip(tick_labels_woot)

    #norm_min = min(refset_norm[:,dim])
    norm_min = 0.0
    #norm_range = np.ptp(refset_norm[:,dim])
    norm_range = 1.0
    norm_step = norm_range/float(ticks-1)
    ticks = [round(norm_min + norm_step*i, 1) for i in range(ticks)]
    ax_i.yaxis.set_ticks(ticks)
    ax_i.set_yticklabels(tick_labels_woot, fontsize=8)
    ax_i.tick_params(axis='y', left=False, right=False)

utilities = ['Watertown', 'Fallsland', 'Dryville', 'Regional']
dv_names = ['RT_W', 'RT_D', 'RT_F', 'TT_D', 'TT_F', 'LMA_W', 'LMA_D', 'LMA_F',\
            'RC_W', 'RC_D', 'RC_F', 'IT_W', 'IT_D', 'IT_F', 'IP_W', 'IP_D', \
            'IP_F', 'INF_W', 'INF_D', 'INF_F']

dv_subset_W = ['RT_W', 'LMA_W', 'RC_W', 'IT_W', 'INF_W']
dv_subset_D = ['RT_D', 'TT_D', 'LMA_D', 'RC_D', 'IT_D', 'INF_D']
dv_subset_F = ['RT_F', 'TT_F', 'LMA_F', 'RC_F', 'IT_F', 'INF_F']

dv_subset_dict = {'W': dv_subset_W, 'D': dv_subset_D, 'F': dv_subset_F}

util_short = ['W', 'D', 'F']
util_full = ['Watertown', 'Dryville', 'Fallsland']

# modify these depending on compsol being analyzed
compSol = 'PW'
compSol_full = 'PW113'
compSol_num = 2

colors_best = ['chocolate', 'rebeccapurple', 'darkgreen']
colors_worst = ['tan', 'mediumpurple', '#36c35c']
colors_danger = ['darkred', 'midnightblue', 'darkolivegreen']

colors_SOS = ['navajowhite', 'thistle', '#beffc1']
comp = ['Fallback\nbargaining', 'Least-squares', 'Power index']

worst_robustness_W = [610, 91, 260]
worst_robustness_D = [543, 433, 250]
worst_robustness_F = [143, 81, 337]
worst_robustness_R = [543, 433, 250]


reg_idx_worst_W = worst_robustness_W[compSol_num]
reg_idx_worst_D = worst_robustness_D[compSol_num]
reg_idx_worst_F = worst_robustness_F[compSol_num]
reg_idx_worst_R = worst_robustness_R[compSol_num]

'''
Get dv filepaths then load the data
'''
dvs_directory = '/home/fs02/pmr82_0001/lbl59/Implementation_Uncertainty/WaterPaths_duReeval/IU_Samples/'
dvs_filepath_p = dvs_directory + 'IU_allMeasures_' + compSol + '.csv'
dvs_filepath_o = dvs_directory + compSol + '_soln.csv'

df_dvs_p = pd.read_csv(dvs_filepath_p, index_col=False, names = dv_names)
df_dvs_p = df_dvs_p.drop(['IP_W', 'IP_D', 'IP_F'], axis=1)
df_dvs_o = pd.read_csv(dvs_filepath_o, index_col=False, names = dv_names)
df_dvs_o = df_dvs_o.drop(['IP_W', 'IP_D', 'IP_F'], axis=1)

df_dvs = pd.concat([df_dvs_p, df_dvs_o], ignore_index=True, axis=0)
df_dvs = df_dvs*100

'''
Get robustness filepaths then load the data
'''
rob_directory = '/home/fs02/pmr82_0001/lbl59/Implementation_Uncertainty/post_processing_du/DU_reeval_output_Apr2022/' \
    + compSol_full + '/'

# load robustness data
filename_robustness = rob_directory + '/robustness_perturbed_og_' + compSol_full + '.csv'
robustness_data = pd.read_csv(filename_robustness, sep=',', names=utilities, index_col=False)
#regional_data = robustness_data['Fallsland']

# identify and list the objectives of the reference set
dv_labels_W = ['Restriction\ntrigger', 'Lake Michael\nallocation', 'Reserve fund\ncontribution', 
                'Insurance\ntrigger', 'Infrastructure\ntrigger']
dv_labels_DF = ['Restriction\ntrigger', 'Transfer\ntrigger', 'Lake Michael\nallocation', 
                 'Reserve fund\ncontribution', 'Insurance\ntrigger', 'Infrastructure\ntrigger']


min_dvs_W = [0.0, 0.0, 0.0, 0.0, 0.0]
min_dvs_DF = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

max_dvs_W = [20.0, 100.0, 20.0, 100.0, 100.0]
max_dvs_DF = [20.0, 70.0, 100.0, 20.0, 100.0, 100.0]

for j in range(len(util_short)):
    parallel_plot_title = util_full[j]
    u = util_short[j]
    utility = utilities[j]
    dv_subset_chosen = dv_subset_dict[u]
    
    robustness_og = robustness_data.iloc[1000,j]
    robustness_SOS_idx = robustness_data[robustness_data[utility] >= robustness_og].index.values
    # sort objectives by regional robustness
    refset_df = df_dvs[dv_subset_chosen]
    refset_sos_df = df_dvs.loc[robustness_SOS_idx, dv_subset_chosen]

    refset = refset_df.to_numpy()
    refset_sos = refset_sos_df.to_numpy()
    
    nrow = refset.shape[0]
    
    # process and normalize the reference set
    min_max_range = {}

    refset_norm = np.copy(refset[:, 0:len(dv_subset_chosen)])
    
    dv_labels = []
    min_dvs = []
    max_dvs = []
    
    if u == 'W':
        dv_labels = dv_labels_W
        min_dvs = min_dvs_W
        max_dvs = max_dvs_W
        reg_idx_worst = reg_idx_worst_W
        
    elif u == 'D':
        dv_labels = dv_labels_DF
        min_dvs = min_dvs_DF
        max_dvs = max_dvs_DF
        reg_idx_worst = reg_idx_worst_D
        
    elif u == 'F':
        dv_labels = dv_labels_DF
        min_dvs = min_dvs_DF
        max_dvs = max_dvs_DF
        reg_idx_worst = reg_idx_worst_F
       
    for i in range(len(dv_labels)):
        
        if dv_labels[i] == 'Lake Michael\nallocation' or dv_labels[i] == 'Reserve fund\ncontribution':
            refset_norm[:,i] = np.true_divide(refset_norm[:,i] - min_dvs[i], (max_dvs[i] - min_dvs[i]))
            min_max_range[dv_labels[i]] = [0, max_dvs[i], (max_dvs[i] - min_dvs[i])]
        
        elif dv_labels[i] == 'Infrastructure\ntrigger':
            refset_norm[:,i] = np.true_divide(100.0 - refset_norm[:,i], (100.0 - 0.0))
            min_max_range[dv_labels[i]] = [0, 100.0, (100.0 - 0.0)]
            
        else:    
            refset_norm[:,i] = np.true_divide(max_dvs[i] - refset_norm[:,i], (max_dvs[i] - min_dvs[i]))  
            min_max_range[dv_labels[i]] = [0, max_dvs[i], (max_dvs[i] - min_dvs[i])]
    
    refset_sos_norm = (refset_norm[robustness_SOS_idx, :])
    
    refset_norm_bottom = refset_norm.min(axis=0)
    refset_norm_top = refset_norm.max(axis=0)
    
    refset_sos_bottom = refset_sos_norm.min(axis=0)
    refset_sos_top = refset_sos_norm.max(axis=0)
    
    # create an array of integers ranging from 0 to the number of objectives
    x = [i for i, _ in enumerate(dv_labels)]
    
    # sharey=False indicates that all the subplot y-axes will be set to different values
    fig, ax  = plt.subplots(1,len(x)-1, sharey=False, figsize=(7,3))
    #fig.set_tight_layout(True)
    # plot all decision variables
    for i, ax_i in enumerate(ax):
        ax_i.fill_between(dv_labels, refset_norm_bottom, refset_norm_top, 
                          color=colors_danger[compSol_num], alpha=0.9, 
                          edgecolor=colors_danger[compSol_num])
        ax_i.tick_params(axis='x', which='major', pad=0.2)
        
    # plot decision variables within SOS
    for i, ax_i in enumerate(ax):
        ax_i.fill_between(dv_labels, refset_sos_bottom, refset_sos_top, 
                          color=colors_SOS[compSol_num], alpha=1.0, label='Safe operating\nspace')
        ax_i.set_xlim([x[i], x[i+1]])
        ax_i.set_xticks([x[i], x[i+1]])
        ax_i.set_xticklabels([dv_labels[i], dv_labels[i+1]], fontsize=8)
        ax_i.tick_params(axis='x', which='major', pad=0.2)
        
    for i, ax_i in enumerate(ax):
        ax_i.plot(dv_labels, refset_norm[1000, :], color=colors_best[compSol_num], alpha=1.0, linewidth=2.5,
                  linestyle='-', label=comp[compSol_num])
        ax_i.grid(False)
        ax_i.set_xlim([x[i], x[i+1]])
        ax_i.set_xticks([x[i], x[i+1]])
        ax_i.set_xticklabels([dv_labels[i], dv_labels[i+1]], fontsize=8)
        ax_i.tick_params(axis='x', which='major', pad=0.2)
        
    for i, ax_i in enumerate(ax):
        ax_i.plot(dv_labels, refset_norm[reg_idx_worst_R, :], color=colors_best[compSol_num], alpha=0.8, linewidth=2.5,
                  linestyle='--', label="Worst regional\nrobustness")
        ax_i.grid(False)
        ax_i.set_xlim([x[i], x[i+1]])
        ax_i.set_xticks([x[i], x[i+1]])
        ax_i.set_xticklabels([dv_labels[i], dv_labels[i+1]], fontsize=8)
        ax_i.tick_params(axis='x', which='major', pad=0.2)
        ax_i.spines['top'].set_visible(False)
        ax_i.spines['bottom'].set_visible(False)
    '''    
    for i, ax_i in enumerate(ax):
        ax_i.plot(dv_labels, refset_norm[reg_idx_best, :], color=colors_best[compSol_num], alpha=0.8, linewidth=2.5,
                  label="Best regional\nrobustness")
        ax_i.grid(False)
        ax_i.set_xlim([x[i], x[i+1]])
        ax_i.set_xticks([x[i], x[i+1]])
        ax_i.set_xticklabels([dv_labels[i], dv_labels[i+1]], fontsize=8)
        ax_i.tick_params(axis='x', which='major', pad=0.2)
        ax_i.spines['top'].set_visible(False)
        ax_i.spines['bottom'].set_visible(False)
    '''    
    for dim, ax_i in enumerate(ax):
        ax_i.xaxis.set_major_locator(ticker.FixedLocator([dim]))
        set_ticks_for_axis(dim, ax_i, ticks=2)
        
    ax2 = plt.twinx(ax[-1])
    dim = len(ax)
    ax2.plot(dv_labels, refset_norm[0, :], color=colors_best[compSol_num], alpha=0.0, linewidth=2.5)
    ax2.xaxis.set_major_locator(ticker.FixedLocator([x[-2], x[-1]]))
    ax2.tick_params(axis='x', which='major', pad=0.2)
    set_ticks_for_axis(dim, ax2, ticks=2)
    ax2.spines['top'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.grid(False)
    ax2.patch.set_alpha(0.01)
    ax2.set_ylim([0,1])
    ax2.set_yticks(np.linspace(ax2.get_yticks()[0], ax2.get_yticks()[-1], len(ax[-1].get_yticks())))
    #mpl_axes_aligner.align.yaxes(ax[-1], 0, ax2, 0, 0.5)
    
    ax2.set_xticklabels([dv_labels[-2], dv_labels[-1]], fontsize=8)
    ax2.tick_params(axis='y', left=False, right=False)
    ax[len(dv_subset_chosen)-2].tick_params(axis='y', left=False, right=False)
    
    plt.subplots_adjust(wspace=0, hspace=0.2, left=0.1, right=0.85, bottom=0.1, top=0.9)
    ax[len(dv_subset_chosen)-2].legend(bbox_to_anchor=(1.0, 1.0), loc='upper right', prop={'size': 6})
    ax[0].set_ylabel("Decision variable\nIncreased use $\longrightarrow$ ", fontsize=9, labelpad=0.9)
    
    '''
    # colorbar
    # creating ScalarMappable
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array([])
    plt.colorbar(sm, pad=0.1)
    '''
    
    plt.suptitle(parallel_plot_title, fontsize=10, y=1.001)
    #plt.tight_layout(w_pad=0)
    fig_filename = 'Figures/parallel_plots/dvs_SOS_' + u + '_' + compSol_full + '.pdf'
    
    plt.savefig(fig_filename, bbox_inches="tight")
    plt.show()
