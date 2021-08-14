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

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
import seaborn as sns
sns.set_style("ticks")
sns.set_context("talk")

# Speficy whether or not to save figures
save_figures = False

# %load_ext autoreload
# %autoreload 2

# %matplotlib notebook

# %%
filename = '/mnt/c/Users/Raeed/data/project-data/smile/cst-gainlag/library/Ford_20180618_COCST_TD.mat'
td = cst.load_clean_data(filename)
# td.set_index('trial_id',inplace=True)

# %%
td['M1_rates'] = [pyaldata.smooth_data(spikes/bin_size,dt=bin_size,std=0.05) 
                  for spikes,bin_size in zip(td['M1_spikes'],td['bin_size'])]
M1_pca_model = sklearn.decomposition.PCA()
td = pyaldata.dim_reduce(td,M1_pca_model,'M1_rates','M1_pca')

# %%
start_time = -0.4
end_time = 0.4
td = pyaldata.restrict_to_interval(
    td,
    start_point_name='idx_goCueTime',
    rel_start=start_time/td.loc[0,'bin_size'],
    rel_end=end_time/td.loc[0,'bin_size']
)

# %%
td_co = td.loc[td['task']=='CO',:].copy()
td_cst = td.loc[td['task']=='CST',:].copy()

target_dirs = td_co['tgtDir'].unique()
dir_colors = plt.get_cmap('Dark2',8)
fig,ax = plt.subplots()
for dirnum,target_dir in enumerate(target_dirs):
    # plot traces
    td_co_dir = td_co.loc[np.isclose(td_co['tgtDir'],target_dir)]
    for neural_trace in td_co_dir['M1_pca'].sample(n=5):
        ax.plot(neural_trace[:,0],neural_trace[:,1],color=dir_colors(dirnum),lw=0.5)
    
    ax.plot(td_co_dir['M1_pca'].mean()[:,0],td_co_dir['M1_pca'].mean()[:,1],color=dir_colors(dirnum),lw=2)
    
cst_trace = td_cst.loc[td_cst['trial_id']==159,'M1_pca'].squeeze()
ax.plot(
    cst_trace[:,0],
    cst_trace[:,1],
    color='k',
    lw=2
)
ax.axis('equal')
ax.axis('off')
