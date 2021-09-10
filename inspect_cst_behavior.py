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
filename = '/mnt/c/Users/Raeed/data/project-data/smile/cst-gainlag/library/Earl_20190716_COCST_TD.mat'
td = cst.load_clean_data(filename)
td.set_index('trial_id',inplace=True)

# %%
td['M1_rates'] = [pyaldata.smooth_data(spikes/bin_size,dt=bin_size,std=0.05) 
                  for spikes,bin_size in zip(td['M1_spikes'],td['bin_size'])]

# %%
# %matplotlib notebook

# subselect CST trials
td_cst = td.loc[td['task']=='CST'].copy()
td_cst = pyaldata.restrict_to_interval(td_cst,start_point_name='idx_cstStartTime',end_point_name='idx_cstEndTime',reset_index=False)
td_cst['trialtime'] = [trial['bin_size']*np.arange(trial['rel_hand_pos'].shape[0]) for _,trial in td_cst.iterrows()]

sm_fig = plt.figure(figsize=(6,6))
gs = mpl.gridspec.GridSpec(4,2,height_ratios=(2,1,1,1))
sm_ax = sm_fig.add_subplot(gs[0,0])
sm_vel_ax = sm_fig.add_subplot(gs[0,1])

@interact(trial_id=list(td_cst.index))
def plot_cst_behavior(trial_id):
    trial = td_cst.loc[trial_id,:]

    sm_scatter_args = {
        'c': trial['trialtime'],
        'cmap': 'viridis',
        's': 5
    }
    
    trial_info = {
        'trialtime': trial['trialtime'],
        'cursor_pos': trial['rel_cursor_pos'][:,0],
        'cursor_vel': trial['cst_cursor_command'][:,0],
        'hand_pos': trial['rel_hand_pos'][:,0],
        'hand_vel': trial['hand_vel'][:,0]
    }
    
    cst.plot_sensorimotor(**trial_info,ax=sm_ax,scatter_args=sm_scatter_args)
    cst.plot_sensorimotor_velocity(**trial_info,ax=sm_vel_ax,scatter_args=sm_scatter_args)
    
    sm_tangent_angle_ax = sm_fig.add_subplot(gs[1,:])
    sm_tangent_magnitude_ax = sm_fig.add_subplot(gs[2,:],sharex=sm_tangent_angle_ax)
    cst.plot_sm_tangent_angle(**trial_info,ax=sm_tangent_angle_ax,scatter_args=sm_scatter_args)
    cst.plot_sm_tangent_magnitude(**trial_info,ax=sm_tangent_magnitude_ax,scatter_args=sm_scatter_args)
    
    hand_energy_ax = sm_fig.add_subplot(gs[3,:],sharex=sm_tangent_angle_ax)
    hand_energy_ax.scatter(
        trial['trialtime'],
        trial['hand_vel'][:,0]**2,
        **sm_scatter_args
    )
    hand_energy_ax.set_ylabel('Hand Energy')
    
#     cursor_energy_ax = sm_fig.add_subplot(gs[4,:],sharex=sm_tangent_angle_ax)
#     cursor_energy_ax.scatter(
#         trial['trialtime'],
#         trial['cst_cursor_command'][:,0]**2,
#         **sm_scatter_args
#     )
#     cursor_energy_ax.set_ylabel('Cursor Energy')


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
td_cst = pyaldata.dim_reduce(td_cst,M1_cst_pca_model,'M1_rates','M1_pca')

behavior_signals = lambda trial: np.column_stack([
    trial['cursor_pos'][:,0],
    trial['hand_pos'][:,0],
    0.25*trial['hand_vel'][:,0]
]).astype(np.float32)
neural_signals = lambda trial: trial['M1_pca'][:,0:3]

cst.plot.make_behavior_neural_widget(td_cst,behavior_signals=behavior_signals,neural_signals=neural_signals)

# %%
# %matplotlib inline

# subselect CST trials
td_cst = td.loc[td['task']=='CST'].copy()
td_cst = pyaldata.restrict_to_interval(td_cst,start_point_name='idx_cstStartTime',end_point_name='idx_cstEndTime',reset_index=False)
td_cst['trialtime'] = [trial['bin_size']*np.arange(trial['hand_pos'].shape[0]) for _,trial in td_cst.iterrows()]

# Pick out specific trial
trial_id = 233
scale=15

trial = td_cst.loc[trial_id,:].squeeze()

sm_scatter_args = {
    'c': 'k',
    's': 2
}

# set up figure layout
plt.rcParams['figure.figsize'] = [10,30]
fig = plt.figure(figsize=(20,10))
gs = mpl.gridspec.GridSpec(3,3)
monitor_ax = fig.add_subplot(gs[0,0])
trace_ax = fig.add_subplot(gs[1:,0],sharex=monitor_ax)
sm_ax = fig.add_subplot(gs[:-1,1])
sm_vel_ax = fig.add_subplot(gs[:-1,2])
# sm_tangent_angle_ax = fig.add_subplot(gs[1,1:])
sm_energy_ax = fig.add_subplot(gs[2,1:])

# monitor view
cursor_sc,hand_sc = cst.plot_cst_monitor_instant(ax=monitor_ax)

# cursor and hand traces (old way)
cursor_l,hand_l = cst.plot_cst_traces(ax=trace_ax,flipxy=True)

# Sensorimotor plot
sm_ax.set_xlim(-scale,scale)
sm_ax.set_ylim(-scale,scale)
sm_sc = cst.plot_sensorimotor(ax=sm_ax,scatter_args=sm_scatter_args,clear_ax=False)

# SM velocity plot
sm_vel_sc = cst.plot_sensorimotor_velocity(ax=sm_vel_ax,scatter_args=sm_scatter_args,clear_ax=False)

# SM tangent angle
# sm_tangent_sc = cst.plot_sm_tangent_angle(ax=sm_tangent_angle_ax,scatter_args=sm_scatter_args)

# SM energy
sm_energy_ax.set_ylim(0,1e4)
sm_energy_sc = cst.plot_sm_tangent_magnitude(ax=sm_energy_ax,scatter_args=sm_scatter_args)
sm_energy_ax.set_ylabel('')

def animate_smplot(i):
    cursor_sc.set_offsets(np.c_[trial['rel_cursor_pos'][i,0],1])
    hand_sc.set_offsets(np.c_[trial['rel_hand_pos'][i,0],-1])
    cursor_l.set_data(trial['rel_cursor_pos'][:i,0],trial['trialtime'][:i])
    hand_l.set_data(trial['rel_hand_pos'][:i,0],trial['trialtime'][:i])
    sm_sc.set_offsets(np.c_[trial['rel_cursor_pos'][:i,0],trial['rel_hand_pos'][:i,0]])
    sm_vel_sc.set_offsets(np.c_[trial['cst_cursor_command'][:i,0],trial['hand_vel'][:i,0]])
    # sm_tangent_sc.set_offsets(np.c_[trial['trialtime'][:i],np.arctan2(trial['hand_vel'][:i,0],trial['cst_cursor_command'][:i,0])*180/np.pi])
    sm_energy_sc.set_offsets(np.c_[trial['trialtime'][:i],trial['hand_vel'][:i,0]**2+trial['cst_cursor_command'][:i,0]**2])
    
ani = mpl.animation.FuncAnimation(
    fig=fig,
    func=animate_smplot,
    interval=30,
    frames=range(0,trial['trialtime'].shape[0],30),
    repeat=False
)

from IPython.display import HTML
HTML(ani.to_jshtml())

# anim_savename = r'/mnt/c/Users/Raeed/Wiki/professional/agendas/smr-meetings/presentations/20210709-cst-analysis/assets/Ford_20180618_CST_trial159_anim.mp4'
# writer = mpl.animation.FFMpegWriter(fps=15) 
# ani.save(anim_savename, writer=writer)


# %%
# _,sm_ax = plt.subplots(1,1,figsize=(5,5))
# sm_ax.plot([-60,60],[60,-60],'--k')
# sm_ax.plot([0,0],[-60,60],'-k')
# sm_ax.plot([-60,60],[0,0],'-k')
# sm_ax.scatter(
#     trial['cursor_pos'][:,0],
#     trial['hand_pos'][:,0],
#     c=trial['trialtime'],
#     cmap='viridis',
#     s=10
# )
# sm_ax.set_xticks([])
# sm_ax.set_yticks([])
# sm_ax.set_xlim(-scale,scale)
# sm_ax.set_ylim(-scale,scale)
# sm_ax.set_xlabel('Cursor position')
# sm_ax.set_ylabel('Hand position')
# sns.despine(ax=sm_ax,left=True,bottom=True)
# # plt.savefig(r'/mnt/c/Users/Raeed/Wiki/professional/cabinet/talks/20210420-ncm2021/assets/Ford_20180618_CST_trial159_smplot.pdf')

# fig,ax = plt.subplots(num_states+1,1,figsize=(6,6),sharex=True,sharey=True)
# trace_ax.hist(
#     trial['hand_vel'][:,0],
#     bins=np.linspace(trial['hand_vel'][:,0].min(),trial['hand_vel'][:,0].max(),30),
#     color='k')
# for statenum in range(num_states):
#     ax[statenum+1].hist(
#         trial['hand_vel'][hmm_z==statenum,0],
#         bins=np.linspace(trial['hand_vel'][:,0].min(),trial['hand_vel'][:,0].max(),30),
#         color=colors[statenum])
#     sns.despine()

# trace_ax.set_title('Hand velocity distribution per state')
# ax[-1].set_xlabel('Hand velocity (mm/s)')
# ax[-1].set_ylabel('Number of 1 ms bins')
# plt.savefig(r'/mnt/c/Users/Raeed/Wiki/professional/cabinet/talks/20210420-ncm2021/assets/Ford_20180618_CST_trial159_handvelhist.pdf')
