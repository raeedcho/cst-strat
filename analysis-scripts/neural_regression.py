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
import k3d
from ipywidgets import interact

# Speficy whether or not to save figures
save_figures = False

# %load_ext autoreload
# %autoreload 2

# %matplotlib notebook

# %%
file_info = {
    'monkey': 'Earl',
    'session_date': '20190716'
}
filename = '/mnt/c/Users/Raeed/data/project-data/smile/cst-gainlag/library/{monkey}_{session_date}_COCST_TD.mat'.format(**file_info)
td = cst.load_clean_data(filename)
td.set_index('trial_id',inplace=True)

# %%
td['M1_rates'] = [pyaldata.smooth_data(spikes/bin_size,dt=bin_size,std=0.05) 
                  for spikes,bin_size in zip(td['M1_spikes'],td['bin_size'])]

# %%
td_cst = td.loc[td['task']=='CST',:].copy()
td_cst = pyaldata.restrict_to_interval(
    td_cst,
    start_point_name='idx_cstStartTime',
    end_point_name='idx_cstEndTime',
    reset_index=False
)

td_co = td.loc[td['task']=='CO',:].copy()
td_co = pyaldata.restrict_to_interval(
    td_co,
    start_point_name='idx_goCueTime',
    # rel_start=-0.4/td_co['bin_size'].values[0],
    rel_end=0.4/td_co['bin_size'].values[0],
    reset_index=False
)

# run PCA
# M1_pca_model = sklearn.decomposition.PCA(n_components=10)
# M1_pca_model.fit(np.concatenate(td_cst['M1_rates'].values))
# td_cst['M1_pca'] = [M1_pca_model.transform(sig) for sig in td_cst['M1_rates']]

# get linear model
horiz_neural_lm = sklearn.linear_model.LinearRegression()
horiz_neural_lm.fit(
    np.concatenate(td_co['M1_rates'].values),
    np.concatenate(td_co['hand_vel'].values),
)
td_co['hand_vel_pred'] = [horiz_neural_lm.predict(neur_sig) for neur_sig in td_co['M1_rates']]
td_cst['hand_vel_pred'] = [horiz_neural_lm.predict(neur_sig) for neur_sig in td_cst['M1_rates']]

# %%
td_co_bin = pyaldata.combine_time_bins(td_co,50)
td_cst_bin = pyaldata.combine_time_bins(td_cst,50)

vel_thresh=80

fig,ax = plt.subplots()
sns.despine(fig, offset=10, trim=True)
@interact(lam=np.unique(td_cst['lambda']))
def scatter_hand_vel_pred(lam):
    # plot it out
    ax.clear()
    ax.plot([-200,300],[vel_thresh,vel_thresh],'--k')
    ax.plot([-200,300],[-vel_thresh,-vel_thresh],'--k')
    ax.scatter(
        np.concatenate(td_co_bin['hand_vel'].values)[:,0],
        np.concatenate(td_co_bin['hand_vel_pred'].values)[:,0],
        c='r',
        s=5
    )
    ax.scatter(
        np.concatenate(td_cst_bin.loc[td_cst['lambda']==lam,'hand_vel'].values)[:,0],
        np.concatenate(td_cst_bin.loc[td_cst['lambda']==lam,'hand_vel_pred'].values)[:,0],
        c='b',
        s=5
    )
    
    ax.axis('equal')
    ax.set_xlabel('True hand velocity')
    ax.set_ylabel('Neural-decoded hand velocity')


# %%
# plot out individual CST trials
fig = plt.figure(figsize=(10,6))
gs = mpl.gridspec.GridSpec(2,6)
neural_ax = fig.add_subplot(gs[0,:3])
behave_ax = fig.add_subplot(gs[1,:3])
sm_ax = fig.add_subplot(gs[:,3:])
    
# plot "neural activity" in top plot
neural_ax.plot([0,6],[0,0],'-k')
neural_ax.plot([0,6],[vel_thresh,vel_thresh],'--k')
neural_ax.plot([0,6],[-vel_thresh,-vel_thresh],'--k')
neural_l, = neural_ax.plot([],[],'-k')
neural_ax.set_xticks([])
neural_ax.set_ylabel('projected activity')
sns.despine(ax=neural_ax,bottom=True,trim=True)

# and hand/cursor movement in bottom
cursor_l,hand_l = cst.plot_cst_traces(ax=behave_ax)
behave_ax.set_ylabel('cursor/hand')

# sm plot on the right
sm_sc = cst.plot_sensorimotor(ax=sm_ax,scatter_args={'c':'k','s':5})
    
@interact(trial_id=list(td_cst.index))
def plot_cst_neural_behavior(trial_id):
    trial = td_cst.loc[trial_id,:]
    trialtime = trial['bin_size']*np.arange(trial['rel_hand_pos'].shape[0])
    
    # plot "neural activity" in top plot
    neural_l.set_data(trialtime,trial['hand_vel_pred'][:,0])
    
    # and hand/cursor movement in bottom
    cursor_l.set_data(trialtime,trial['rel_hand_pos'][:,0])
    hand_l.set_data(trialtime,trial['hand_vel'][:,0])
    
    # sm plot on the right
    cst.plot_sensorimotor(
        ax=sm_ax,
        cursor_pos=trial['rel_cursor_pos'][:,0],
        hand_pos = trial['rel_hand_pos'][:,0],
        scatter_args={
            's':5,
            'c': abs(trial['hand_vel_pred'][:,0]),
            'cmap': 'viridis'
        }
    )
#     sm_sc.set_offsets(np.column_stack([
#         trial['rel_cursor_pos'][:,0],
#         trial['rel_hand_pos'][:,0]
#     ]))
#     sm_sc.set_color(abs(trial['hand_vel_pred']))
