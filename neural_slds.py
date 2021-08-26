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
import autograd.numpy as np
import autograd.numpy.random as npr

import pandas as pd
import cst
import pyaldata
import scipy

import ssm
from ssm.util import find_permutation
from ssm.plots import gradient_cmap, white_to_color_cmap

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
# %matplotlib notebook

import seaborn as sns
sns.set_style("ticks")
sns.set_context("talk")

import k3d

from ipywidgets import interact

color_names = [
    "dusty purple",
    "faded green",
    "orange",
    "amber",
    "windows blue"
    ]

colors = sns.xkcd_palette(color_names)
cmap = gradient_cmap(colors)


# Speficy whether or not to save figures
save_figures = False

# %%
# filename = '/data/raeed/project-data/smile/cst-gainlag/library/Ford_20180618_COCST_TD.mat'
filename = '/mnt/c/Users/Raeed/data/project-data/smile/cst-gainlag/library/Ford_20180618_COCST_TD.mat'
td = cst.load_clean_data(filename)
td.set_index('trial_id',inplace=True)

# %%
## Try switching model on CO data first
td_co = td.loc[td['task']=='CO',:].copy()
td_co = pyaldata.restrict_to_interval(
    td_co,
    start_point_name='idx_goCueTime',
    rel_start=-0.4/td_co['bin_size'].values[0],
    rel_end=0.4/td_co['bin_size'].values[0],
    reset_index=False
)
td_co = pyaldata.combine_time_bins(td_co,20)

# SLDS params
emissions_dim = td_co['M1_spikes'].values[0].shape[1]
num_disc_states = 2 # hold, prep, and move? Seemingly only seems to use 2 states anyway (hold and move)
latent_dim = 6 # just a guess

# extract neural data + inputs
neural_datas = list(td_co['M1_spikes'])
go_cue = [np.float32(np.arange(sig.shape[0])==go_cue_idx) for sig,go_cue_idx in zip(td_co['M1_spikes'],td_co['idx_goCueTime'])]

# fit the SLDS
slds = ssm.SLDS(emissions_dim,num_disc_states,latent_dim,emissions='poisson_orthog')
elbos,q = slds.fit(neural_datas,num_iters=20,alpha=0.0)

# add states back into data structure
td_co['M1_slds_cont'] = q.mean_continuous_states
td_co['M1_slds_disc'] = [slds.most_likely_states(cont_latent,data) for cont_latent,data in zip(td_co['M1_slds_cont'],td_co['M1_spikes'])]

# make plots
fig,ax = plt.subplots()
ax.plot(elbos, label="Laplace EM")
ax.set_xlabel("EM Iteration")
ax.set_ylabel("ELBO")
ax.legend(loc="lower right")

# %%
# plot out discrete and continuous latents for each given trial
fig,ax = plt.subplots(4,1,figsize=(8,8))

@interact(trial_id=list(td_co.index))
def plot_slds_fit(trial_id):
    trial = td_co.loc[trial_id,:]
    
    for gca in ax:
        gca.clear()
    
    cmap_limited = mpl.colors.ListedColormap(colors[0:num_disc_states])
    ax[0].imshow(trial['M1_slds_disc'][None,:],aspect='auto',cmap=cmap_limited)
    ax[0].plot(trial['idx_goCueTime']*np.array([1,1]),[-0.5,0.5],'--r')
    ax[0].set_yticks([])
    ax[0].set_xticks([])
    ax[0].set_title('Most likely discrete state')
    
    sns.despine(ax=ax[0],left=True,bottom=True)

    lim = abs(trial['M1_slds_cont']).max()
    for d in range(latent_dim):
        ax[1].plot(trial['M1_slds_cont'][:,d],'-k')
    ax[1].plot(trial['idx_goCueTime']*np.array([1,1]),lim*np.array([-1,1]),'--r')
    # ax[1].set_yticks(np.arange(latent_dim) * lim)
    # ax[1].set_yticklabels(["$x_{}$".format(d+1) for d in range(latent_dim)])
    ax[1].set_xticks([])
    ax[1].set_title('Estimated continuous latents')
    sns.despine(ax=ax[1],bottom=True,trim=True)

    ax[2].imshow(trial['M1_spikes'].T,aspect='auto',cmap='inferno')
    ax[2].plot(trial['idx_goCueTime']*np.array([1,1]),[0,trial['M1_spikes'].shape[1]],'--r')
    ax[2].set_xticks([])
    ax[2].set_yticks([])
    ax[2].set_title('Emissions raster')
    sns.despine(ax=ax[1],bottom=True,trim=True)

    hand_pos = trial['hand_pos']-[0,470,0]
    lim = abs(hand_pos[:2]).max()
    for d in range(2):
        ax[3].plot(hand_pos[:,d]+lim*d,'-k')
    ax[3].plot(trial['idx_goCueTime']*np.array([1,1]),lim*np.array([-1,1]),'--r')
    ax[3].set_yticks(np.arange(2) * lim)
    ax[3].set_yticklabels(['hand x','hand y'])
    ax[3].set_title('Hand position')
    sns.despine(ax=ax[3],trim=True)



# %%
target_dirs = td_co['tgtDir'].unique()
dir_colors = plt.get_cmap('Dark2',8)
dims_to_plot=[1,2,3]
# fig,ax = plt.subplots()
trace_plot = k3d.plot(name='Neural Traces')
for dirnum,target_dir in enumerate(target_dirs):
    # plot traces
    color_val = int(255*dir_colors(dirnum)[0]) << 16 | int(255*dir_colors(dirnum)[1]) << 8 | int(255*dir_colors(dirnum)[2])
    td_co_dir = td_co.loc[np.isclose(td_co['tgtDir'],target_dir)]
    for neural_trace in td_co_dir['M1_slds_cont'].sample(n=5):
        trace_plot+=k3d.line(neural_trace[:,dims_to_plot],shader='mesh',width=0.01,color=color_val)
    
    trace_plot+=k3d.line(td_co_dir['M1_slds_cont'].mean()[:,dims_to_plot],shader='mesh',width=0.1,color=color_val)

trace_plot.display()
# ax.axis('equal')
# ax.axis('off')

# %%

# %%
# try a decoder on new continuous state?
M1_slds_lm = sklearn.linear_model.LinearRegression()
M1_slds_lm.fit(
    np.concatenate(td_co['M1_slds_cont'].values),
    np.concatenate(td_co['hand_vel'].values),
)
td_co['slds_pred'] = [M1_slds_lm.predict(neur_sig) for neur_sig in td_co['M1_slds_cont']]

td_co['M1_rates'] = [pyaldata.smooth_data(spikes/bin_size,dt=bin_size,std=0.05) 
                  for spikes,bin_size in zip(td_co['M1_spikes'],td_co['bin_size'])]
M1_rates_lm = sklearn.linear_model.LinearRegression()
M1_rates_lm.fit(
    np.concatenate(td_co['M1_rates'].values),
    np.concatenate(td_co['hand_vel'].values),
)
td_co['rates_pred'] = [M1_rates_lm.predict(neur_sig) for neur_sig in td_co['M1_rates']]

# %%
vel_thresh=80

fig,ax = plt.subplots()

# plot it out
ax.plot([-200,300],[vel_thresh,vel_thresh],'--k')
ax.plot([-200,300],[-vel_thresh,-vel_thresh],'--k')
ax.scatter(
    np.concatenate(td_co['hand_vel'].values)[:,0],
    np.concatenate(td_co['rates_pred'].values)[:,0],
    c='k',
    s=5
)
ax.scatter(
    np.concatenate(td_co['hand_vel'].values)[:,0],
    np.concatenate(td_co['slds_pred'].values)[:,0],
    c='b',
    s=5
)
ax.axis('equal')
ax.set_xlabel('True hand velocity')
ax.set_ylabel('Neural-decoded hand velocity')
sns.despine(fig, offset=10, trim=True)

# %%
