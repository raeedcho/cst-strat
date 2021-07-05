# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.3
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

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
import seaborn as sns
sns.set_style("ticks")
sns.set_context("talk")
# %matplotlib inline

# Speficy whether or not to save figures
save_figures = False

# %%
# filename = '/data/raeed/project-data/smile/cst-gainlag/library/python/Ford_20180618_COCST_TD.mat'
filename = '/mnt/c/Users/Raeed/data/project-data/smile/cst-gainlag/library/python/Ford_20180618_COCST_TD.mat'
td = pyaldata.mat2dataframe(filename,True,'td_cst')
td.set_index('trial_id',inplace=True)

# %%
import importlib
importlib.reload(cst)

from ipywidgets import interact

@interact(trial_id=list(td.index))
def plot_cst_trial(trial_id):
    trial = td.loc[trial_id,:]

    sm_scatter_args = {
        'c': trial['trialtime'],
        'cmap': 'viridis',
        's': 10
    }

#     # old way
#     trace_fig,trace_ax = plt.subplots(2,1,figsize=(7.5,5),sharex=True)
#     trace_ax[0].plot([0,6],[0,0],'-k')
#     trace_ax[0].plot(trial['trialtime'],trial['cursor_pos'][:,0],'-b')
#     trace_ax[0].plot(trial['trialtime'],trial['hand_pos'][:,0],'-r')
#     trace_ax[0].set_xlim(0,6)
#     trace_ax[0].set_xticks([])
#     trace_ax[0].set_ylim(-60,60)
#     trace_ax[0].set_ylabel('Cursor or hand position')
#     trace_ax[0].set_title('$\lambda = {}$'.format(trial['lambda']))
    
#     trace_ax[1].plot([0,6],[0,0],'-k')
#     trace_ax[1].plot(trial['trialtime'],trial['cst_cursor_command'][:,0],'-b')
#     trace_ax[1].plot(trial['trialtime'],trial['hand_vel'][:,0],'-r')
#     trace_ax[1].set_xlim(0,6)
#     trace_ax[1].set_xticks([])
#     trace_ax[1].set_ylim(-100,100)
#     trace_ax[1].set_ylabel('Cursor or hand velocity')
#     trace_ax[1].set_xlabel('Time (s)')
#     sns.despine(ax=trace_ax[0],left=False,bottom=True,trim=True)
#     sns.despine(ax=trace_ax[1],left=False,trim=True)
    sm_fig = plt.figure(figsize=(6,6))
    gs = mpl.gridspec.GridSpec(4,2,height_ratios=(2,1,1,1))
    sm_ax = sm_fig.add_subplot(gs[0,0])
    sm_vel_ax = sm_fig.add_subplot(gs[0,1])
    cst.plot_sensorimotor(trial,ax=sm_ax,scatter_args=sm_scatter_args)
    cst.plot_sensorimotor_velocity(trial,ax=sm_vel_ax,scatter_args=sm_scatter_args)
    
    sm_tangent_angle_ax = sm_fig.add_subplot(gs[1,:])
    sm_tangent_magnitude_ax = sm_fig.add_subplot(gs[2,:],sharex=sm_tangent_angle_ax)
    cst.plot_sm_tangent_angle(trial,ax=sm_tangent_angle_ax,scatter_args=sm_scatter_args)
    cst.plot_sm_tangent_magnitude(trial,ax=sm_tangent_magnitude_ax,scatter_args=sm_scatter_args)
    
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
# Pick out specific trial
trial_id = 159
scale=35

trial = td.loc[trial_id,:]

# set up figure layout
plt.rcParams['figure.figsize'] = [10,30]
fig = plt.figure(figsize=(10,5))
gs = mpl.gridspec.GridSpec(3,3)
monitor_ax = fig.add_subplot(gs[0,0])
trace_ax = fig.add_subplot(gs[1:,0],sharex=monitor_ax)
sm_ax = fig.add_subplot(gs[:-1,1])
sm_vel_ax = fig.add_subplot(gs[:-1,2])
# sm_tangent_angle_ax = fig.add_subplot(gs[1,1:])
sm_energy_ax = fig.add_subplot(gs[2,1:],sharex=sm_tangent_angle_ax)

# monitor view
monitor_ax.fill_between(
    [-50,50],
    0,y2=2,
    color=[0.5,0.5,0.5]
)
monitor_ax.set_xlim(-60,60)
monitor_ax.set_ylim(-2,2)
monitor_ax.set_xticks([])
monitor_ax.set_yticks([])
cursor_sc = monitor_ax.scatter(0,1,s=60,c='b')
hand_sc = monitor_ax.scatter(0,-1,s=60,c='r')
sns.despine(ax=monitor_ax,left=True,bottom=True)

# cursor and hand traces (old way)
trace_ax.plot([0,0],[0,6],'-k')
trace_ax.set_ylim(0,6)
trace_ax.set_xlim(-60,60)
trace_ax.set_xticks([-50,50])
trace_ax.set_yticks([])
trace_ax.set_ylabel('Time (s)')
trace_ax.set_xlabel('Cursor or hand position')
sns.despine(ax=trace_ax,left=True,bottom=False,trim=True)
cursor_l, = trace_ax.plot([],[],'-b')
hand_l, = trace_ax.plot([],[],'-r')

# Sensorimotor plot
sm_ax.set_xlim(-scale,scale)
sm_ax.set_ylim(-scale,scale)
sm_ax.set_xticks([])
sm_ax.set_yticks([])
sm_ax.yaxis.tick_right()
sm_ax.set_xlabel('Cursor position')
sm_ax.set_ylabel('Hand position')
sns.despine(ax=sm_ax,left=True,bottom=True)
sm_ax.plot([-60,60],[60,-60],'--k')
sm_ax.plot([0,0],[-60,60],'-k')
sm_ax.plot([-60,60],[0,0],'-k')
sm_l, = sm_ax.plot([],[],'k',lw=4)

# SM velocity plot
sm_vel_ax.set_xlim(-60,60)
sm_vel_ax.set_ylim(-100,100)
sm_vel_ax.set_xticks([])
sm_vel_ax.set_yticks([])
sm_vel_ax.yaxis.tick_right()
sm_vel_ax.set_xlabel('Cursor velocity')
sm_vel_ax.set_ylabel('Hand velocity')
sns.despine(ax=sm_vel_ax,left=True,bottom=True)
sm_vel_ax.plot([-60,60],[60,-60],'--g')
sm_vel_ax.plot([0,0],[-100,100],'--k')
sm_vel_ax.plot([-60,60],[0,0],'-k')
sm_vel_l, = sm_vel_ax.plot([],[],'k',lw=4)


# SM tangent angle
# sm_tangent_angle_ax.plot(
#     [trial['trialtime'][0],trial['trialtime'][-1]],
#     [-90,-90],
#     '--k'
# )
# sm_tangent_angle_ax.plot(
#     [trial['trialtime'][0],trial['trialtime'][-1]],
#     [90,90],
#     '--k'
# )
# sm_tangent_angle_ax.plot(
#     [trial['trialtime'][0],trial['trialtime'][-1]],
#     [-45,-45],
#     '--g'
# )
# sm_tangent_angle_ax.plot(
#     [trial['trialtime'][0],trial['trialtime'][-1]],
#     [135,135],
#     '--g'
# )
# sm_tangent_angle_ax.fill_between(
#     trial['trialtime'],
#     0, y2=90,
#     color=[1,0.8,0.8]
# )
# sm_tangent_angle_ax.fill_between(
#     trial['trialtime'],
#     -180, y2=-90,
#     color=[1,0.8,0.8]
# )
# # sm_tangent_angle_ax.set_ylabel('SM tangent angle')
# sm_tangent_angle_ax.set_xticks([])
# sm_tangent_angle_ax.set_yticks([-180,-90,0,90,180])
# sns.despine(ax=sm_tangent_angle_ax,left=False,trim=True)
# sm_tangent_l = sm_tangent_angle_ax.scatter([],[],s=5,c='k')

# SM energy
sm_energy_ax.set_xlabel('Time (s)')
# sm_energy_ax.set_ylabel('SM tangent magnitude')
sm_energy_ax.set_xticks(np.arange(7))
sm_energy_ax.set_ylim(0,1e4)
sns.despine(ax=sm_energy_ax,left=False,trim=True)
sm_energy_l, = sm_energy_ax.plot([],[],'-k')

def animate_smplot(i):
    cursor_sc.set_offsets(np.c_[trial['cursor_pos'][i,0],1])
    hand_sc.set_offsets(np.c_[trial['hand_pos'][i,0],-1])
    cursor_l.set_data(trial['cursor_pos'][:i,0],trial['trialtime'][:i])
    hand_l.set_data(trial['hand_pos'][:i,0],trial['trialtime'][:i])
    sm_l.set_data(trial['cursor_pos'][:i,0],trial['hand_pos'][:i,0])
    sm_vel_l.set_data(trial['cst_cursor_command'][:i,0],trial['hand_vel'][:i,0])
    # sm_tangent_l.set_offsets(np.c_[trial['trialtime'][:i],np.arctan2(trial['hand_vel'][:i,0],trial['cst_cursor_command'][:i,0])*180/np.pi])
    sm_energy_l.set_data(trial['trialtime'][:i],trial['hand_vel'][:i,0]**2+trial['cst_cursor_command'][:i,0]**2)
    
ani = mpl.animation.FuncAnimation(
    fig=fig,
    func=animate_smplot,
    interval=30,
    frames=range(0,trial['trialtime'].shape[0],30),
    repeat=False
)

from IPython.display import HTML
HTML(ani.to_jshtml())

# anim_savename = r'/mnt/c/Users/Raeed/Wiki/professional/cabinet/talks/20210420-ncm2021/assets/Ford_20180618_CST_trial159_anim.mp4'
# writer = mpl.animation.FFMpegWriter(fps=30) 
# ani.save(anim_savename, writer=writer)

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

