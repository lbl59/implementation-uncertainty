import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

'''
Plot radial barplots
Adapted from: https://www.python-graph-gallery.com/web-heatmap-and-radial-barchart-plastics

@author: lbl59
'''

sns.set_theme(style="whitegrid")

'''
Applies styles and customizations to a circular axis object
'''
def style_polar_axis(ax, x_data):
    #ax.set_theta_offset(np.pi/2)
    #ax.set_theta_direction(-1)
    #ax.set_frame_on(False)
       
    ax.set_xticklabels([])
    ax.set_ylim(0, len(x_data))
    ax.set_yticks(np.arange(0, len(x_data)))
    ax.set_yticklabels([])
    ax.grid(alpha=0.4)
    return ax 

'''
Take a circular axis for a utility and its assigned color.
Adds labels corresponding to each decision variable
'''
def add_labels_polar_axis(ax, color, x_data):
    bbox_dict = {
        "facecolor": "w", "edgecolor": color, "linewidth": 0.5,
        "boxstyle": "round", "pad": 0.15}

    for idx, x in enumerate([x_data.reverse()]):
        ax.text(0, idx, x, color=color, ha='center', va='center',
                fontsize=12, bbox=bbox_dict)

    return ax

'''
Creates the radial plot
'''
def plot_circular(axes, x_data, y_utils, sensitivity_values, segment_col):
    axes_flattened = axes.ravel()

    for u in range(len(y_utils)):
        # Select data for given objective
        rob_sensitivity_sorted = np.sort(sensitivity_values.iloc[u,:len(x_data)].values)[::-1]
        rob_sensitivity = np.flip(rob_sensitivity_sorted[:5])
        ax = axes_flattened[u]
        radii = 10 * rob_sensitivity
        #width = np.pi / 4 * len(x_data)
        width = 4*np.pi / len(x_data)
        ax.set_ylabel(y_utils[u], loc='center', size=8)
        
        ax = style_polar_axis(ax, x_data[:5])
        #proportions = rob_sensitivity * (2*np.pi)
        #y_pos = np.arange(len(rob_sensitivity))

        #x = np.linspace(0, proportions, num=200)
        #y = np.vstack([y_pos] * 200)
        color = segment_col[u]
        theta = np.linspace(0.0, 2 * np.pi, 5, endpoint=False)
        #ax.plot(x, y, lw=4, color=color, solid_capstyle='round')
        ax.bar(theta, radii, width=width, bottom=0.0, color=color, alpha=0.7)
        
        ax.set_title(y_utils[u], pad=10, color="k")

        ax = add_labels_polar_axis(ax, color, x_data)
        ax.grid(b=True, which='major', color='darkgrey', linestyle='-')
    return axes

# change depending on compromise solution and whether its sensitivity to DUF or DVs
compSol_full = 'PW113'
mode = 'DV'

LS_colors = ['darkviolet', 'darkviolet', 'darkviolet', 'darkviolet']
PW_colors = ['forestgreen', 'forestgreen', 'forestgreen', 'forestgreen']

segment_colors = {'LS98': LS_colors, 'PW113': PW_colors}

segment_col = segment_colors[compSol_full]

main_dir = '/home/fs02/pmr82_0001/lbl59/Implementation_Uncertainty/post_processing_du/delta_output/'
robustness_filepath = main_dir + 'delta_robustness_' + mode + '/all_utils_robustness_' + compSol_full + '.csv'
robustness = pd.read_csv(robustness_filepath, index_col=False)

sensitivity_values = robustness

savefig_dir = '/home/fs02/pmr82_0001/lbl59/Implementation_Uncertainty/post_processing_du/Figures/sensitivity_radial/'
savefig_name = savefig_dir + 'robustness_' + compSol_full + '_' + mode + '.pdf'

grid_kws = {"width_ratios": (0.25,0.25,0.25,0.25), "wspace": 0.2}
f, axes = plt.subplots(1, 4, figsize=(8, 3), subplot_kw={"projection": "polar"}, gridspec_kw=grid_kws)
plt.subplots_adjust(top = 0.95, bottom = 0.05,
            hspace = 0.1, wspace = 0.1)

y_utils = ['Watertown', 'Dryville', 'Fallsland', 'Regional']

x_dvs=['$RT_{W}$', '$RT_{D}$', '$RT_{F}$', '$TT_{D}$', '$TT_{F}$', '$LM_{W}$',
       '$LM_{D}$', '$LM_{F}$', '$RC_{W}$', '$RC_{D}$', '$RC_{F}$', '$IT_{W}$',
       '$IT_{D}$', '$IT_{F}$', '$INF_{W}$', '$INF_{D}$', '$INF_{F}$']

x_dufs = ['WRE', 'DRE', 'FRE', 'DMP', 'BTM', 'BIM', 'IIM', 'STM', 'SFM',
          'SPM', 'EMP', 'PT', 'CT']
x_data = []

#ax1 = fig.add_subplot(411)
title = ''
if mode == 'DUF':
    title = 'Sensitivity of robustness to deeply-uncertain factors'
    x_data = x_dufs
elif  mode == 'DV':
    title = 'Sensitivity of robustness to decision variables'
    x_data = x_dvs


plot_circular(axes, x_data, y_utils, sensitivity_values, segment_col)

plt.savefig(savefig_name)
plt.show()
