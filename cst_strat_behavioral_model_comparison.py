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
import autograd.numpy as np
import autograd.numpy.random as npr

import sklearn

import pandas as pd
import pyaldata

import ssm

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
# %matplotlib inline

from tqdm.notebook import tqdm

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
cmap = ssm.plots.gradient_cmap(colors)

# Speficy whether or not to save figures
save_figures = False

# %%
# filename = '/data/raeed/project-data/smile/cst-gainlag/library/python/Ford_20180618_COCST_TD.mat'
filename = '/mnt/c/Users/Raeed/data/project-data/smile/cst-gainlag/library/python/Ford_20180618_COCST_TD.mat'
# td = cst.get_cst_dataframe(filename)
td = pyaldata.mat2dataframe(filename,True,'td_cst')
td.set_index('trial_id',inplace=True)

# %%
# subselect a table of specific lambda
lambda_to_use = 3.3
td_lambda = td[td['lambda']==lambda_to_use]

# %%
# set up cross validation over a set of models
num_states_list = [1,2,4,5]
model_params_df = pd.DataFrame(
    [{
        'model_id':modelnum,
        'num_iters':50,
        'num_states':num_states,
        'obs_dim':1,
        'input_dims':3,
        'observations':'input_driven_gaussian',
        'transitions':'recurrent_only',
    } for modelnum,num_states in enumerate(num_states_list)]
)

def evaluate_hmm(td,train_idx,test_idx,model_params):
    full_obs = [el[:,0][:,None] for el in td['hand_vel']]
    full_input = [np.column_stack((pos[:,0],vel[:,0],hand_pos[:,0]))
                for pos,vel,hand_pos in zip(td['cursor_pos_shift'],td['cursor_vel_shift'],td['hand_pos'])]
    
    train_obs = [full_obs[i] for i in train_idx]
    test_obs = [full_obs[i] for i in test_idx]
    train_input = [full_input[i] for i in train_idx]
    test_input = [full_input[i] for i in test_idx]

    # hand_vel = A_{z_t}*(hand_vel_{t-1}) + V_{z_t}*[cursor_pos_shift;cursor_vel_shift;hand_pos] + b_{z_t} + \omega
    hmm = ssm.HMM(
        int(model_params['num_states']),
        int(model_params['obs_dim']),
        M=int(model_params['input_dims']),
        observations=str(model_params['observations']),
        transitions=str(model_params['transitions']),
    )
    # hmm = ssm.HMM(num_states, obs_dim, M=input_dims, observations="autoregressive",transitions='recurrent_only',observation_kwargs=dict(l2_penalty_A=1e10))

    hmm_lls = hmm.fit(
        train_obs,
        inputs=train_input,
        method="em",
        num_iters=int(model_params['num_iters']),
        init_method="kmeans", #can also use random for initialization method, which sometimes works better
        verbose=0,
    )
    
    return hmm.log_probability(test_obs,inputs=test_input)

# run crossvalidation loop
kf = sklearn.model_selection.KFold(n_splits=5,shuffle=True)
model_ll_list = []
for foldnum,(train_idx,test_idx) in tqdm(enumerate(kf.split(td_lambda)), total=kf.get_n_splits(), desc="k-fold"):
    for _,model_params in tqdm(
        model_params_df.iterrows(),
        total=model_params_df.shape[0],
        position=1,
        leave=False,
        desc="parameter sets"):
        
        model_ll_list.append(
            {
                'crossval_id': foldnum,
                'model_id': model_params['model_id'],
                'log_prob': evaluate_hmm(td_lambda,train_idx,test_idx,model_params)
            }
        )
        
model_lls = pd.DataFrame(model_ll_list)
print(model_lls)


# %%
print(pd.merge(model_lls,model_params_df,on='model_id',how='left').groupby('model_id')[['num_states','log_prob']].mean())
# print(model_lls.join(model_params_df).groupby('model_id').mean())
