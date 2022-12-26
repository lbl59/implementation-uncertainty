import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme()

# change depending on compromise solution and whether its sensitivity to DUF or DVs
compSol_full = 'LS98'
cmap_col = 'Purples'
mode = 'DV'
rot = 45    # if DV use 0; if DUF use 45

print('mode: ', mode)

main_dir = '/home/fs02/pmr82_0001/lbl59/Implementation_Uncertainty/post_processing_du/delta_output/'
robustness_filepath = main_dir + 'delta_robustness_' + mode + '/all_utils_robustness_' + compSol_full + '.csv'
robustness = pd.read_csv(robustness_filepath, index_col=False)

savefig_dir = '/home/fs02/pmr82_0001/lbl59/Implementation_Uncertainty/post_processing_du/Figures/sensitivity_heatmaps/'
savefig_name = savefig_dir + 'robustness_' + mode + '_' + compSol_full + '.pdf'

grid_kws = {"height_ratios": (0.90, .05), "hspace": 0.4}
f, (ax1, cbar_ax) = plt.subplots(2, figsize=(15, 5), gridspec_kw=grid_kws)
plt.subplots_adjust(top = 0.95, bottom = 0.05,
            hspace = 0, wspace = 0.05)

y_objs = ['Watertown', 'Dryville', 'Fallsland', 'Regional']
x_dvs=['$RT_{W}$', '$RT_{D}$', '$RT_{F}$', '$TT_{D}$', '$TT_{F}$', '$LM_{W}$', 
       '$LM_{D}$', '$LM_{F}$', '$RC_{W}$', '$RC_{D}$', '$RC_{F}$', '$IT_{W}$', 
       '$IT_{D}$', '$IT_{F}$', '$IN_{W}$', '$IN_{D}$', '$IN_{F}$']

x_dufs = ['WRE', 'DRE', 'FRE', 'DMP', 'BTM', 'BIM', 'IIM', 'STM', 'SFM', 
          'SPM', 'EMP', '$CRR_{L}$ PT', '$CRR_{L}$ CT', '$CRR_{H}$ PT', '$CRR_{H}$ CT', 
          'WR1 PT', 'WR1 CT', 'WR2 PT', 'WR2 CT', 'DR PT', 'DR CT', 'FR PT', 'FR CT']
x_data = []
plt.rc('xtick', labelsize=5)
plt.rc('ytick', labelsize=5)
plt.rc('axes', labelsize=9)
plt.rc('axes', titlesize=10)

#ax1 = fig.add_subplot(411)
title = ''
if mode == 'DUF':
    title = 'Sensitivity of robustness to deeply-uncertain factors'
    x_data = x_dufs
elif  mode == 'DV':
    title = 'Sensitivity of robustness to decision variables'
    x_data = x_dvs
    
ax1.set_title(title, size=14)     # change depending on whether analyzing DUF or DV
ax1 = sns.heatmap(robustness, linewidths=.05, cmap=cmap_col, xticklabels=x_data, yticklabels=y_objs, ax=ax1, cbar=True,
                  cbar_ax = cbar_ax, cbar_kws = {'orientation': 'horizontal'})
ax1.set_yticklabels(y_objs, rotation=0)
ax1.set_xticklabels(x_data, rotation=rot, ha='center')

cbar_kws={'orientation': 'horizontal'}
plt.savefig(savefig_name)
plt.show()
