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
# %matplotlib notebook

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
    gs = mpl.gridspec.GridSpec(3,2,height_ratios=(2,1,1))
    cst.plot_sensorimotor(trial,ax=sm_fig.add_subplot(gs[0,0]),scatter_args=sm_scatter_args)
    cst.plot_sensorimotor_velocity(trial,sm_fig.add_subplot(gs[0,1]),scatter_args=sm_scatter_args)
    
    cst.plot_sm_tangent_angle(trial,ax=sm_fig.add_subplot(gs[1,:]),scatter_args=sm_scatter_args)
    cst.plot_sm_tangent_magnitude(trial,ax=sm_fig.add_subplot(gs[2,:]),scatter_args=sm_scatter_args)

# %%
# Pick out specific trial
trial_id = 159
color_by_state=True
scale=35

trial = td_lambda.loc[trial_id,:]
data = trial['hand_vel'][:,0][:,None]
inpt = np.column_stack((trial['cursor_pos_shift'][:,0],trial['cursor_vel_shift'][:,0],trial['hand_pos'][:,0]))
hmm_z = hmm.most_likely_states(data,input=inpt)

# old fig + SM plots
fig,ax = plt.subplots(1,2,figsize=(10,5))
ax[0].plot([0,6],[0,0],'-k')
ax[0].set_xlim(0,6)
ax[0].set_ylim(-scale,scale)
ax[0].set_yticks([-30,30])
ax[0].set_xticks([])
ax[0].set_xlabel('Time (s)')
ax[0].set_ylabel('Cursor or hand position')
sns.despine(ax=ax[0],left=False,bottom=True,trim=True)
ax[1].set_xlim(-scale,scale)
ax[1].set_ylim(-scale,scale)
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[1].yaxis.tick_right()
ax[1].set_xlabel('Cursor position')
ax[1].set_ylabel('Hand position')
sns.despine(ax=ax[1],left=True,bottom=True)
ax[1].plot([-60,60],[60,-60],'--k')
ax[1].plot([0,0],[-60,60],'-k')
ax[1].plot([-60,60],[0,0],'-k')

cursor_l, = ax[0].plot([],[],'-b')
hand_l, = ax[0].plot([],[],'-r')
sm_l, = ax[1].plot([],[],'k',lw=4)

def animate_smplot(i):
    cursor_l.set_data(trial['trialtime'][:i],trial['cursor_pos'][:i,0])
    hand_l.set_data(trial['trialtime'][:i],trial['hand_pos'][:i,0])
    sm_l.set_data(trial['cursor_pos'][:i,0],trial['hand_pos'][:i,0])
    
ani = mpl.animation.FuncAnimation(
    fig=fig,
    func=animate_smplot,
    interval=30,
    frames=range(0,trial['trialtime'].shape[0],30),
    repeat=False
)

# from IPython.display import HTML
# HTML(ani.to_jshtml())

# anim_savename = r'/mnt/c/Users/Raeed/Wiki/professional/cabinet/talks/20210420-ncm2021/assets/Ford_20180618_CST_trial159_anim.mp4'
# writer = mpl.animation.FFMpegWriter(fps=30) 
# ani.save(anim_savename, writer=writer)

_,sm_ax = plt.subplots(1,1,figsize=(5,5))
sm_ax.plot([-60,60],[60,-60],'--k')
sm_ax.plot([0,0],[-60,60],'-k')
sm_ax.plot([-60,60],[0,0],'-k')
sm_ax.scatter(
    trial['cursor_pos'][:,0],
    trial['hand_pos'][:,0],
    c=hmm_z,
    cmap=cmap,
    s=10,
    norm=mpl.colors.Normalize(vmin=0,vmax=len(color_names)-1)
)
sm_ax.set_xticks([])
sm_ax.set_yticks([])
sm_ax.set_xlim(-scale,scale)
sm_ax.set_ylim(-scale,scale)
sm_ax.set_xlabel('Cursor position')
sm_ax.set_ylabel('Hand position')
sns.despine(ax=sm_ax,left=True,bottom=True)
# plt.savefig(r'/mnt/c/Users/Raeed/Wiki/professional/cabinet/talks/20210420-ncm2021/assets/Ford_20180618_CST_trial159_smplot.pdf')

fig,ax = plt.subplots(num_states+1,1,figsize=(6,6),sharex=True,sharey=True)
ax[0].hist(
    trial['hand_vel'][:,0],
    bins=np.linspace(trial['hand_vel'][:,0].min(),trial['hand_vel'][:,0].max(),30),
    color='k')
for statenum in range(num_states):
    ax[statenum+1].hist(
        trial['hand_vel'][hmm_z==statenum,0],
        bins=np.linspace(trial['hand_vel'][:,0].min(),trial['hand_vel'][:,0].max(),30),
        color=colors[statenum])
    sns.despine()

ax[0].set_title('Hand velocity distribution per state')
ax[-1].set_xlabel('Hand velocity (mm/s)')
ax[-1].set_ylabel('Number of 1 ms bins')
# plt.savefig(r'/mnt/c/Users/Raeed/Wiki/professional/cabinet/talks/20210420-ncm2021/assets/Ford_20180618_CST_trial159_handvelhist.pdf')

