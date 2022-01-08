# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import pandas as pd
import scipy
import pyaldata
import cst
import sklearn
import ssa
import elephant.gpfa

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
import seaborn as sns
sns.set_style("ticks")
sns.set_context("talk")
import k3d

from ipywidgets import interact

# Speficy whether or not to save figures
save_figures = False

# %load_ext autoreload
# %autoreload 2

# %%
# filename = '/data/raeed/project-data/smile/cst-gainlag/library/python/Ford_20180618_COCST_TD.mat'
filename = '/mnt/c/Users/Raeed/data/project-data/smile/cst-gainlag/library/Ford_20180618_COCST_TD.mat'
td = cst.load_clean_data(filename)
td.set_index('trial_id',inplace=True)

# %%
td['M1_rates'] = [pyaldata.smooth_data(spikes/bin_size,dt=bin_size,std=0.05) 
                  for spikes,bin_size in zip(td['M1_spikes'],td['bin_size'])]

# %%
# test out SSA against PCA
td_task = td.loc[td['task']=='CO',:].copy()
td_task = pyaldata.restrict_to_interval(
    td_task,
    start_point_name='idx_goCueTime',
    rel_start=-400,
    rel_end=400,
    reset_index=False
)
td_task['trialtime'] = [trial['bin_size']*np.arange(trial['hand_pos'].shape[0]) for _,trial in td_task.iterrows()]

td_task_select = td_task.copy()

# parameters
num_dims = 10

# fit PCA model
M1_task_pca_model = sklearn.decomposition.PCA()
M1_task_pca_model.fit(np.concatenate(td_task_select['M1_rates'].values))
td_task_select['M1_pca'] = [M1_task_pca_model.transform(sig) for sig in td_task_select['M1_rates']]
# td_task_select = pyaldata.dim_reduce(td_task_select,M1_task_pca_model,'M1_rates','M1_pca')

# fit SSA model (start with PCA because it's mean centered and it's the initialization anyway)
M1_task_ssa_model = cst.models.SSA(R=num_dims,n_epochs=3000,lr=0.01)
M1_task_ssa_model.fit(np.concatenate(td_task_select['M1_rates'].values))
td_task_select['M1_ssa'] = [M1_task_ssa_model.transform(sig) for sig in td_task_select['M1_rates']]
# td_task_select = pyaldata.dim_reduce(td_task_select,M1_task_ssa_model,'M1_pca','M1_ssa')

# fit GPFA model


# %%
trial_to_plot = td_task_select.index[0]

plt.figure(figsize=(10,5))
for i in range(num_dims):
    
    # Plot SSA results
    plt.subplot(num_dims,2,2*i+1)
    plt.plot(td_task_select.loc[trial_to_plot,'trialtime'][[0,-1]],[0,0],color='k')
    plt.plot(td_task_select.loc[trial_to_plot,'trialtime'],td_task_select.loc[trial_to_plot,'M1_ssa'][:,i])
    
#     plt.ylim([-1.1, 1.1])
    plt.yticks([])    
    if i<num_dims-1:
        plt.xticks([])
    else:
        plt.xlabel('Time')

    # Plot PCA results
    plt.subplot(num_dims,2,2*i+2)
    plt.plot(td_task_select.loc[trial_to_plot,'trialtime'][[0,-1]],[0,0],color='k')
    plt.plot(td_task_select.loc[trial_to_plot,'trialtime'],td_task_select.loc[trial_to_plot,'M1_pca'][:,i])
    
#     plt.ylim([-1.1, 1.1])
    plt.yticks([])
    if i<num_dims-1:
        plt.xticks([])
    else:
        plt.xlabel('Time')

#Titles
plt.subplot(num_dims,2,1)
plt.title('SSA LowD Projections')

plt.subplot(num_dims,2,2)
plt.title('PCA LowD Projections')
