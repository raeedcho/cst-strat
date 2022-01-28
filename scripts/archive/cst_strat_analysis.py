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
# %matplotlib widget

import seaborn as sns
sns.set_style("ticks")
sns.set_context("talk")

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

# %% Load data
file_query = {
    'monkey': 'Earl',
    'session_date': '20190716'
}
td = cst.load_clean_data(file_query)

# %% Extract and condition CST data for HMM analysis
# restrict to time points of interest
td_cst = td.loc[td['task']=='CST',:].copy()
td_cst = pyaldata.restrict_to_interval(
    td_cst,
    start_point_name='idx_cstStartTime',
    end_point_name='idx_cstEndTime',
    reset_index=False
)

## Get smoothed pca
td_cst['M1_rates'] = [pyaldata.smooth_data(spikes/bin_size,dt=bin_size,std=0.05) 
                  for spikes,bin_size in zip(td_cst['M1_spikes'],td_cst['bin_size'])]
M1_pca_model = sklearn.decomposition.PCA()
td_cst = pyaldata.dim_reduce(td_cst,M1_pca_model,'M1_rates','M1_pca')

# combine bins to get 10 ms bins
td_cst = pyaldata.combine_time_bins(td_cst,10)

# %% Fit HMM to CST data
# lambda_to_use = 3.3
# td_lambda = td[td['lambda']==lambda_to_use]
# td_lambda = td_cst
# control_obs = [el[:,0][:,None] for el in td_lambda['hand_vel']]
hmm_obs = [el[:,0][:,None] for el in td_cst['hand_vel']]
# hmm_input = [pos[:,0][:,None] for pos in td_cst['rel_hand_pos']]
# hmm_input = [np.column_stack((pos[:,0],vel[:,0],hand_pos[:,0]))
#             for pos,vel,hand_pos in zip(td_lambda['cursor_pos'],td_lambda['cst_cursor_command'],td_lambda['hand_pos'])]

N_iters = 50
num_states = 3
obs_dim = 1
input_dims = 0

# hand_vel = A_{z_t}*(hand_vel_{t-1}) + V_{z_t}*[cursor_pos_shift;cursor_vel_shift;hand_pos] + b_{z_t} + \omega
hmm = ssm.HMM(num_states, obs_dim, M=input_dims, observations="gaussian",transitions="standard")
# hmm = ssm.HMM(num_states, obs_dim, M=input_dims, observations="autoregressive",transitions='recurrent_only',observation_kwargs=dict(l2_penalty_A=1e10))
# hmm = ssm.HMM(num_states, obs_dim, observations="autoregressive",transitions='sticky')

# hmm_lls = hmm.fit(hmm_obs, inputs=hmm_input, method="em", num_iters=N_iters, init_method="kmeans") #can also use random for initialization method, which sometimes works better
hmm_lls = hmm.fit(hmm_obs, method="em", num_iters=N_iters, init_method="kmeans") #can also use random for initialization method, which sometimes works better

# make plots for HMM log likelihood over training
plt.plot(hmm_lls, label="EM")
plt.xlabel("EM Iteration")
plt.ylabel("Log Probability")
plt.legend(loc="lower right")
plt.show()

# permute states to keep things consistent, order states by average velocities
state_order = np.argsort(np.squeeze(hmm.observations.mus))
hmm.permute(state_order)

# insert hmm most likely states into trialdata
td_cst['hmm_state'] = [hmm.most_likely_states(vel[:,0][:,None],input=pos[:,0][:,None])
                       for vel,pos in zip(td_cst['hand_vel'],td_cst['rel_hand_pos'])]

# %%
from ipywidgets import interact
# %matplotlib notebook

trace_fig, trace_ax = plt.subplots(3, 1, figsize=(7.5, 5), sharex=True)
sm_fig, sm_ax = plt.subplots(1, 2, figsize=(10, 5))
hist_fig, hist_ax = plt.subplots(num_states+1, 1, figsize=(6, 6), sharex=True, sharey=True)

@interact(trial_id=list(td_cst.index),scale=(10,60),color_by_state=True)
def plot_cst_trial(trial_id,scale,color_by_state):
    trial = td_cst.loc[trial_id,:].copy()
    trial['trialtime'] = trial['bin_size']*np.arange(trial['rel_hand_pos'].shape[0])
    data = trial['hand_vel'][:,0][:,None]
#     inpt = np.column_stack((trial['cursor_pos_shift'][:,0],trial['cst_cursor_command_shift'][:,0],trial['hand_pos'][:,0]))
    
    # Plot the true and inferred discrete states
    hmm_z = trial['hmm_state']
    posterior_probs = hmm.expected_states(data=data, input=trial['rel_hand_pos'])[0]
    
    if color_by_state:
        sm_scatter_args = {
            'c': hmm_z,
            'cmap': cmap,
            's': 10,
            'norm': mpl.colors.Normalize(vmin=0,vmax=len(color_names)-1)
        }
    else:
        sm_scatter_args = {
            'c': 'k',
            'cmap': 'viridis',
            's': 10
        }

    # old way
    trace_ax[0].clear()
    trace_ax[0].plot([0,6],[0,0],'-k')
    trace_ax[0].plot(trial['trialtime'],trial['rel_cursor_pos'][:,0],'-b')
    trace_ax[0].plot(trial['trialtime'],trial['rel_hand_pos'][:,0],'-r')
    trace_ax[0].set_xlim(0,6)
    trace_ax[0].set_xticks([])
    trace_ax[0].set_ylim(-scale,scale)
    trace_ax[0].set_ylabel('Cursor or hand position')
    trace_ax[0].set_title('$\lambda = {}$'.format(trial['lambda']))
    
    trace_ax[1].clear()
    trace_ax[1].plot([0,6],[0,0],'-k')
#     trace_ax[1].plot(trial['trialtime'],trial['M1_pca'][:,0],'-c')
#     trace_ax[1].plot(trial['trialtime'],trial['M1_pca'][:,1],'-m')
#     trace_ax[1].plot(trial['trialtime'],trial['M1_pca'][:,2],'-y')
#     trace_ax[1].plot(trial['trialtime'],trial['M1_pca'][:,3],'-k')
    trace_ax[1].plot(trial['trialtime'],trial['cst_cursor_command'][:,0],'-b')
    trace_ax[1].plot(trial['trialtime'],trial['hand_vel'][:,0],'-r')
    trace_ax[1].set_xlim(0,6)
    trace_ax[1].set_xticks([])
    trace_ax[1].set_ylim(-100,100)
    trace_ax[1].set_ylabel('Neural state')
    
    
    trace_ax[2].clear()
    for k in range(num_states):
        trace_ax[2].plot(trial['trialtime'],posterior_probs[:, k], label="State " + str(k + 1), lw=2,
             color=colors[k])
    trace_ax[2].set_ylim((-0.01, 1.01))
    trace_ax[2].set_yticks([0, 0.5, 1])
    trace_ax[2].set_xlim(0,6)
    trace_ax[2].set_xlabel('Time (s)')
    trace_ax[2].set_ylabel("p(state)")
    
    sns.despine(ax=trace_ax[0],left=False,bottom=True,trim=True)
    sns.despine(ax=trace_ax[1],left=False,bottom=True,trim=True)
    sns.despine(ax=trace_ax[2],left=False,trim=True)
    
    cst.plot_sensorimotor(
        cursor_pos=trial['rel_cursor_pos'][:,0],
        hand_pos=trial['rel_hand_pos'][:,0],
        ax=sm_ax[0],
        scatter_args=sm_scatter_args
    )
    sm_ax[0].set_xlim(-scale,scale)
    sm_ax[0].set_ylim(-scale,scale)
    cst.plot_sensorimotor_velocity(
        cursor_vel=trial['cst_cursor_command'][:,0],
        hand_vel=trial['hand_vel'][:,0],
        ax=sm_ax[1],
        scatter_args=sm_scatter_args
    )
#     sm_ax[2].scatter(
#         trial['M1_pca'][:,0],
#         trial['M1_pca'][:,1],
#         **sm_scatter_args
#     )
    
#     cst.plot_sm_tangent_polar(trial,scatter_args=sm_scatter_args)
    
    for ax in hist_ax:
        ax.clear()
    hist_ax[0].hist(
        trial['hand_vel'][:,0],
        bins=np.linspace(trial['hand_vel'][:,0].min(),trial['hand_vel'][:,0].max(),30),
        color='k')
    for statenum in range(num_states):
        hist_ax[statenum+1].hist(
            trial['hand_vel'][hmm_z==statenum,0],
            bins=np.linspace(trial['hand_vel'][:,0].min(),trial['hand_vel'][:,0].max(),30),
            color=colors[statenum])
        sns.despine()
        
    hist_ax[0].set_title('Hand velocity distribution per state')
    hist_ax[-1].set_xlabel('Hand velocity (mm/s)')
    hist_ax[-1].set_ylabel('Number of 1 ms bins')


# %%
for i in range(num_states):
    plt.subplot(4,2,2*i+2)
    plt.imshow(hmm.observations.mus[i],aspect='auto',cmap='RdBu',clim=[-1,1])
    if i==0:
        plt.title('Predicted')
    plt.ylabel('State'+str(i))

# %%
def plot_most_likely_dynamics(model,
    xlim=(-30, 30), ylim=(-100, 100), nxpts=20, nypts=20,
    alpha=0.8, ax=None, figsize=(3, 3)):
    
    K = model.K
    assert model.D == 2
    x = np.linspace(*xlim, nxpts)
    y = np.linspace(*ylim, nypts)
    X, Y = np.meshgrid(x, y)
    xy = np.column_stack((X.ravel(), Y.ravel()))

    # Get the probability of each state at each xy location
    z = np.argmax(xy.dot(model.transitions.Rs.T) + model.transitions.r, axis=1)

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

    for k, (A, b) in enumerate(zip(model.dynamics.As, model.dynamics.bs)):
        dxydt_m = xy.dot(A.T) + b - xy

        zk = z == k
        if zk.sum(0) > 0:
            ax.quiver(xy[zk, 0], xy[zk, 1],
                      dxydt_m[zk, 0], dxydt_m[zk, 1],
                      color=colors[k % len(colors)], alpha=alpha)

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')

    plt.tight_layout()

    return ax


# %%
print(hmm.transitions.Rs)
print(hmm.transitions.r)
print(hmm.transitions.Ws)

# %%
fig,ax = plt.subplots(1,1)
ax.imshow(np.squeeze(hmm.transitions.transition_matrices(np.ones((3,1)),input=np.zeros((3,2)),mask=None,tag=None)))

# %%
np.unique(td['lambda'])
lambda_to_use = 3.3
td_lambda = td[td['lambda']==lambda_to_use]
control_obs = [el[:,0][:,None] for el in td_lambda['hand_pos']]
vis_input = [np.column_stack((pos[:,0],vel[:,0]))
            for pos,vel in zip(td_lambda['cursor_pos_shift'],td_lambda['cursor_vel_shift'])]
for trial_id,trial in td_lambda.iterrows():
    display(trial)

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

# %%
# find neural covariance matrices in each state and plot against each other
full_state_list = pd.Series(np.concatenate(td_cst['hmm_state'].values))
full_neural_state = pd.DataFrame(np.concatenate(td_cst['M1_rates'].values))

state_corr = [full_neural_state.loc[full_state_list==state,:].corr().values.flatten() for state in range(num_states)]
state_corr = pd.DataFrame(state_corr).T

pd.plotting.scatter_matrix(state_corr,alpha=0.2)

# check subspace alignment index
def subspace_alignment(ref_cov,eig_vecs,eig_vals):
    return np.trace(eig_vecs.T @ ref_cov @ eig_vecs)/sum(eig_vals)

state_covs = [full_neural_state.loc[full_state_list==state,:].cov().values for state in range(num_states)]
state_eigs = [np.linalg.eig(cov) for cov in state_covs]

align_index = np.zeros((num_states,num_states))
for ref_state in range(num_states):
    for check_state in range(num_states):
        align_index[ref_state,check_state] = subspace_alignment(
            state_covs[ref_state],
            state_eigs[check_state][1][:,0:10],
            state_eigs[check_state][0][0:10]
        )
        
## Answers to these questions--neural manifold is quite aligned across all states

# %%
# What does the neural repertoire look like during the "wait" state?
M1_hold_pca_model = sklearn.decomposition.PCA(n_components=10)
M1_hold_pca_model.fit(full_neural_state.loc[full_state_list==1,:].values)

hold_neural_state = M1_hold_pca_model.transform(full_neural_state.loc[full_state_list==1,:].values)

fig,ax = plt.subplots()
@interact(trial_id = list(td_cst.index))
def plot_hold_neural_state(trial_id):
    hold_neural_state = M1_hold_pca_model.transform(td_cst.loc[trial_id,'M1_rates'])
    ax.clear()
    ax.scatter(
        hold_neural_state[:,0],
        hold_neural_state[:,1],
        alpha=0.2,
        c=td_cst.loc[trial_id,'hmm_state'],
        cmap=cmap,
        norm=mpl.colors.Normalize(vmin=0,vmax=len(color_names)-1)
    )
