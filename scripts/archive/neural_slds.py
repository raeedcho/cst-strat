# %% Importing and setup
import autograd.numpy as np
import autograd.numpy.random as npr

import pandas as pd
import cst
import pyaldata
import scipy
import sklearn

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

# %load_ext autoreload
# %autoreload 2
# %autosave 0

# Speficy whether or not to save figures
save_figures = False

# %%
file_query = {
    'monkey': 'Earl',
    'session_date': '20190716'
}
td = cst.load_clean_data(file_query)

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
elbos,q = slds.fit(neural_datas,num_iters=40,alpha=0.0)

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
fig = plt.figure(figsize=(8,8))

# @interact(trial_id=list(td_co.index))
def plot_slds_fit(trial_id):
    trial = td_co.loc[trial_id,:]
    
    fig.clear()   
    [beh_ax,disc_ax,cont_ax,emis_ax] = fig.subplots(4,1)
    
    hand_pos = trial['rel_hand_pos']
    lim = abs(hand_pos[:2]).max()
    for d in range(2):
        beh_ax.plot(hand_pos[:,d]+lim*d,'-k')
    beh_ax.plot(trial['idx_goCueTime']*np.array([1,1]),lim*np.array([-1,1]),'--r')
    beh_ax.set_xticks([])
    beh_ax.set_yticks(np.arange(2) * lim)
    beh_ax.set_yticklabels(['horizontal','vertical'])
    beh_ax.set_title('Hand position')
    sns.despine(ax=beh_ax,trim=True,bottom=True)
    
    cmap_limited = mpl.colors.ListedColormap(colors[0:num_disc_states])
    disc_ax.imshow(trial['M1_slds_disc'][None,:],aspect='auto',cmap=cmap_limited)
    disc_ax.plot(trial['idx_goCueTime']*np.array([1,1]),[-0.5,0.5],'--r')
    disc_ax.set_yticks([])
    disc_ax.set_xticks([])
    disc_ax.set_title('Most likely discrete state')
    
    sns.despine(ax=disc_ax,left=True,bottom=True)

    lim = abs(trial['M1_slds_cont']).max()
    for d in range(latent_dim):
        cont_ax.plot(trial['M1_slds_cont'][:,d],'-k')
    cont_ax.plot(trial['idx_goCueTime']*np.array([1,1]),lim*np.array([-1,1]),'--r')
    # cont_ax.set_yticks(np.arange(latent_dim) * lim)
    # cont_ax.set_yticklabels(["$x_{}$".format(d+1) for d in range(latent_dim)])
    cont_ax.set_xticks([])
    cont_ax.set_title('Estimated continuous latents')
    sns.despine(ax=cont_ax,bottom=True,trim=True)

    emis_ax.imshow(trial['M1_spikes'].T,aspect='auto',cmap='inferno')
    emis_ax.plot(trial['idx_goCueTime']*np.array([1,1]),[0,trial['M1_spikes'].shape[1]],'--r')
    emis_ax.set_yticks([])
    emis_ax.set_title('Emissions raster')
    sns.despine(ax=emis_ax,trim=True)

# plot_slds_fit(23)

# %%
target_dirs = td_co['tgtDir'].unique()
dir_colors = plt.get_cmap('Dark2',8)
dims_to_plot=[0,4,5]
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
## Now try switching model on CST data
td_cst = td.loc[td['task']=='CST',:].copy()
td_cst = pyaldata.restrict_to_interval(
    td_cst,
    start_point_name='idx_cstStartTime',
    rel_start=-0.2/td_cst['bin_size'].values[0],
    end_point_name='idx_cstEndTime',
    reset_index=False
)
td_cst = pyaldata.combine_time_bins(td_cst,20)
# td_cst = pyaldata.soft_normalize_signal(td_cst,'M1_spikes',alpha=5)

# SLDS params
emissions_dim = td_cst['M1_spikes'].values[0].shape[1]
num_disc_states = 3 # Complete guess
latent_dim = 10 # just a guess

# extract neural data + inputs
neural_datas = list(td_cst['M1_spikes'])
go_cue = [np.float32(np.arange(sig.shape[0])==go_cue_idx) for sig,go_cue_idx in zip(td_cst['M1_spikes'],td_cst['idx_goCueTime'])]

# fit the SLDS
slds_cst = ssm.SLDS(emissions_dim,num_disc_states,latent_dim,emissions='poisson_orthog')
elbos_cst,q_cst = slds_cst.fit(neural_datas,num_iters=40,alpha=0.0)

# add states back into data structure
td_cst['M1_slds_cont'] = q_cst.mean_continuous_states
td_cst['M1_slds_disc'] = [slds_cst.most_likely_states(cont_latent,data) for cont_latent,data in zip(td_cst['M1_slds_cont'],td_cst['M1_spikes'])]

# make plots
fig,ax = plt.subplots()
ax.plot(elbos_cst, label="Laplace EM")
ax.set_xlabel("EM Iteration")
ax.set_ylabel("ELBO")
ax.legend(loc="lower right")

# %%
# plot out discrete and continuous latents for each given trial
fig = plt.figure(figsize=(8,8))
fig,

@interact(trial_id=list(td_cst.index))
def plot_slds_fit(trial_id):
    trial = td_cst.loc[trial_id,:].copy()
    
    trial['trialtime']=trial['bin_size']*np.arange(trial['hand_pos'].shape[0])
    
    fig.clear()
    [beh_ax,disc_ax,cont_ax,emis_ax] = fig.subplots(4,1,sharex=True)
    
    cursor_l,hand_l = cst.plot_cst_traces(
        ax=beh_ax,
        cursor_pos=trial['rel_cursor_pos'][:,0],
        hand_pos=trial['rel_hand_pos'][:,0],
        trialtime=trial['trialtime']
    )
    lim = abs(trial['rel_hand_pos'][:,0]).max()
#     beh_ax.plot(trial['idx_goCueTime']*np.array([1,1]),lim*np.array([-1,1]),'--r')
    
    cmap_limited = mpl.colors.ListedColormap(colors[0:num_disc_states])
    disc_ax.imshow(
        trial['M1_slds_disc'][None,:],
        aspect='auto',
        cmap=cmap_limited,
        extent=(0,trial['trialtime'][-1],0.5,-0.5))
#     disc_ax.plot(trial['trialtime'][trial['idx_goCueTime']]*np.array([1,1]),[-0.5,0.5],'--r')
    disc_ax.set_yticks([])
    disc_ax.set_xticks([])
    disc_ax.set_title('Most likely discrete state')
    sns.despine(ax=disc_ax,left=True,bottom=True)

    for d in range(latent_dim):
        cont_ax.plot(trial['trialtime'],trial['M1_slds_cont'][:,d],'-k')
    lim = abs(trial['M1_slds_cont']).max()
#     cont_ax.plot(trial['idx_goCueTime']*np.array([1,1]),lim*np.array([-1,1]),'--r')
    cont_ax.set_xticks([])
    cont_ax.set_title('Estimated continuous latents')
    sns.despine(ax=cont_ax,bottom=True,trim=True)

    emis_ax.imshow(
        trial['M1_spikes'].T,
        aspect='auto',
        cmap='inferno',
        extent=(0,trial['trialtime'][-1],0.5,-0.5),
    )
#     emis_ax.plot(trial['idx_goCueTime']*np.array([1,1]),[0,trial['M1_spikes'].shape[1]],'--r')
    emis_ax.set_xticks([])
    emis_ax.set_yticks([])
    emis_ax.set_title('Emissions raster')
    emis_ax.set_xticks(np.arange(6)+1)
    sns.despine(ax=cont_ax,bottom=True,trim=True)


