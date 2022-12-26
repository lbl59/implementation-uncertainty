
import numpy as np
import os
import pandas as pd
from pandas.plotting import parallel_coordinates
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns

sns.set_theme(style="white")

'''
Plot all regional objectives that meet satisficing criteria and highlight
the least-squares (Social Planner) and power index (Pragmatist) compromises

@author: lbl59

'''

def set_ticks_for_axis(dim, ax_i, ticks):
    min_val, max_val, v_range = min_max_range[obj_names[dim]]

    step = v_range/float(ticks-1)
    
    
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
    norm_min = min(refset_norm[:,dim])
    norm_range = np.ptp(refset_norm[:,dim])
    norm_step =(norm_range/float(ticks-1))
    ticks = [round(norm_min + norm_step*i, 1) for i in range(ticks)]
    ax_i.yaxis.set_ticks(ticks)
    ax_i.set_yticklabels(tick_labels_woot, fontsize=8)
    ax_i.tick_params(axis='y', left=False, right=False)
    #ax_i.set_ylim([min_val, max_val])
    
objs_directory = '/home/fs02/pmr82_0001/lbl59/Implementation_Uncertainty/FB_data/'
objs_filename = 'Trindade_reference_set_objectives.csv'

# load data
refset = np.loadtxt(objs_directory+objs_filename, delimiter=",", skiprows=1)
refset = refset[:, 1:]

#print(refset.shape)
refset = np.append(refset, [refset[171,:]], axis=0)
refset = np.append(refset, [refset[113,:]], axis=0)
refset = np.append(refset, [refset[98,:]], axis=0)

#print(refset.shape)
nrow = refset.shape[0]
# identify and list the objectives of the reference set
obj_names = ['Reliability', 'Restriction\nfrequency', 'Infrastructure\nnet present cost ($mil)', 
             'Peak\nfinancial cost', 'Worst-case\ncost']
refset[:,[0,1,3,4]] = refset[:,[0,1,3,4]]*100
# create an array of integers ranging from 0 to the number of objectives
x = [i for i, _ in enumerate(obj_names)]

# sharey=False indicates that all the subplot y-axes will be set to different values
fig, ax  = plt.subplots(1,len(x)-1, sharey=False, figsize=(12,3))

min_max_range = {}
refset_norm = np.copy(refset)

for i in range(len(obj_names)):
    if i == 0:
        refset_norm[:,i] = np.true_divide(max(refset_norm[:,i]) - refset_norm[:,i], np.ptp(refset_norm[:,i]))
    else:
        refset_norm[:,i] = np.true_divide(refset_norm[:,i] - min(refset_norm[:,i]), np.ptp(refset_norm[:,i]))

    min_max_range[obj_names[i]] = [min(refset[:,i]), max(refset[:,i]), np.ptp(refset[:,i])]


for i, ax_i in enumerate(ax):
    for d in range(len(refset)):
        if(d == 1):
            ax_i.plot(obj_names, refset_norm[d, :], color='lightgrey', alpha=0.7, label='Other', linewidth=3)
        else:
            ax_i.plot(obj_names, refset_norm[d, :], color='lightgrey', alpha=0.7, linewidth=3)

for i, ax_i in enumerate(ax):
    for d in range(len(refset)):
        if (d == (nrow-3)):
            #ax_i.plot(obj_names, refset_norm[d, :], color='orange', alpha=0.0, linewidth=4, label="Fallback bargaining")
            ax_i.plot(obj_names, refset_norm[d, :], color='orange', alpha=0.0, linewidth=4)
        elif (d == (nrow-2)):
            ax_i.plot(obj_names, refset_norm[d, :], color='darkgreen', alpha=1.0, linewidth=4, label="Power index")
        elif (d == (nrow-1)):
            ax_i.plot(obj_names, refset_norm[d, :], color='indigo', alpha=1.0, linewidth=4, label="Least-squares")

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
ax2.set_xticklabels([obj_names[-2], obj_names[-1]])
ax2.tick_params(axis='y', left=False, right=False)
ax2.spines['top'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.tick_params(axis='x', which='major', pad=0.2)
ax[3].tick_params(axis='y', left=False, right=False)
ax[3].tick_params(axis='x', which='major', pad=0.2)
ax[3].spines['top'].set_visible(False)
ax[3].spines['bottom'].set_visible(False)
plt.subplots_adjust(wspace=0, hspace=0.2, left=0.1, right=0.85, bottom=0.1, top=0.9)
ax[3].legend(bbox_to_anchor=(1.05, 0.95), loc='upper left', prop={'size': 10})
ax[0].set_ylabel("$\leftarrow$ Direction of preference", fontsize=10)
ax[0].spines['top'].set_visible(False)
ax[0].spines['bottom'].set_visible(False)
plt.suptitle("Original performance objectives", fontsize=10, y=1.001)
  
plt.savefig("Figures/parallel_plots/compSol_refsets.pdf")
plt.show()
