# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.12.0
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
import k3d
from ipywidgets import interact

# Speficy whether or not to save figures
save_figures = False

# %load_ext autoreload
# %autoreload 2

# %matplotlib notebook

# %%
filename = '/mnt/c/Users/Raeed/data/project-data/smile/cst-gainlag/library/Ford_20180618_COCST_TD.mat'
td = cst.load_clean_data(filename)
td.set_index('trial_id',inplace=True)

# %%
td['M1_rates'] = [pyaldata.smooth_data(spikes/bin_size,dt=bin_size,std=0.05) 
                  for spikes,bin_size in zip(td['M1_spikes'],td['bin_size'])]
M1_pca_model = sklearn.decomposition.PCA()
td = pyaldata.dim_reduce(td,M1_pca_model,'M1_rates','M1_pca')

# %%
# GPFA (elephant keeps killing kernel...)
# new_bin_size = 0.02
# M1_gpfa_model = cst.models.GPFA(bin_size=new_bin_size,x_dim=10)
# M1_gpfa_model.fit(td['M1_spikes'].to_list())
# M1_gpfa = M1_gpfa_model.transform(td['M1_spikes'].to_list())
# td_bin = pyaldata.combine_time_bins(td,np.floor(new_bin_size/td.loc[td.index[0],'bin_size']))
# td_bin['M1_gpfa'] = M1_gpfa

# %%
start_time = -0.4
end_time = 0.4
td_trim = pyaldata.restrict_to_interval(
    td,
    start_point_name='idx_goCueTime',
    rel_start=start_time/td.loc[td.index[0],'bin_size'],
    rel_end=end_time/td.loc[td.index[0],'bin_size'],
    reset_index=False
)

# %%
td_co = td_trim.loc[td_trim['task']=='CO',:].copy()
td_cst = td_trim.loc[td_trim['task']=='CST',:].copy()

target_dirs = td_co['tgtDir'].unique()
dir_colors = plt.get_cmap('Dark2',8)
# fig,ax = plt.subplots()
trace_plot = k3d.plot(name='Neural Traces')
for dirnum,target_dir in enumerate(target_dirs):
    # plot traces
    color_val = int(255*dir_colors(dirnum)[0]) << 16 | int(255*dir_colors(dirnum)[1]) << 8 | int(255*dir_colors(dirnum)[2])
    td_co_dir = td_co.loc[np.isclose(td_co['tgtDir'],target_dir)]
    for neural_trace in td_co_dir['M1_pca'].sample(n=5):
        trace_plot+=k3d.line(neural_trace[:,0:3],shader='mesh',width=0.2,color=color_val)
    
    trace_plot+=k3d.line(td_co_dir['M1_pca'].mean()[:,0:3],shader='mesh',width=1,color=color_val)
    
trace_plot+=k3d.line(
    td_cst.loc[159,'M1_pca'][:,0:3],
    shader='mesh',
    width=1,
    color=0,
)
trace_plot.display()
# ax.axis('equal')
# ax.axis('off')

# %%
td_cst = td.loc[td['task']=='CST',:].copy()
td_cst = pyaldata.restrict_to_interval(
    td_cst,
    start_point_name='idx_cstStartTime',
    end_point_name='idx_cstEndTime',
    reset_index=False
)
td_cst['trialtime'] = [trial['bin_size']*np.arange(trial['hand_pos'].shape[0]) for _,trial in td_cst.iterrows()]
M1_cst_pca_model = sklearn.decomposition.PCA()
td_cst = pyaldata.dim_reduce(td_cst,M1_cst_pca_model,'M1_rates','M1_cst_pca')


cst_trace_plot = k3d.plot(name='CST neural traces',camera_fov=40)
cst_trace_line = k3d.line(
    np.array([0,0,0],dtype=np.float32),
    attribute=np.float32(6),
    color_map=k3d.matplotlib_color_maps.Viridis,
    color_range=[0,6],
    shader='mesh',
    width=0.25,
)
cst_trace_dot = k3d.points(
    np.array([0,0,0],dtype=np.float32),
    color=0,
    point_size=2
)
cst_trace_plot+=cst_trace_line
cst_trace_plot+=cst_trace_dot
cst_trace_plot.display()
cst_trace_plot.start_auto_play()

@interact(trial_id=list(td_cst.index))
def plot_cst_neural_trace(trial_id):
    trial = td_cst.loc[trial_id,:]
    # plot traces
    cst_trace_line.vertices = trial['M1_cst_pca'][:,0:3]
    cst_trace_line.attribute = trial['trialtime'].astype(np.float32)
    # cst_trace_dot.positions = {
    #     str(time*5): neural_state
    #     for time,neural_state in zip(trial['trialtime'][0:-1:50],trial['M1_cst_pca'][0:-1:50,0:3])
    # }
    cst_trace_dot.positions = trial['M1_cst_pca'][0,0:3]

