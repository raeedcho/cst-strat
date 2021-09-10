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
import pyaldata
import cst

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
sns.set_style("ticks")
sns.set_context("talk")
from ipywidgets import interact

# Speficy whether or not to save figures
save_figures = False

# %load_ext autoreload
# %autoreload 2
# %autosave 0

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
td_bin = pyaldata.combine_time_bins(td,10)
# recalculate speed after binning
td_bin['hand_speed'] = [np.linalg.norm(vel,axis=1) for vel in td_bin['hand_vel']]

td_cst = td_bin.loc[td_bin['task']=='CST',:].copy()
td_co = td_bin.loc[td_bin['task']=='CO',:].copy()

hold_slice_fun = pyaldata.generate_epoch_fun(
        start_point_name='idx_goCueTime',
        rel_start = -0.3/td_bin['bin_size'].values[0],
        rel_end = 0,
        )
move_slice_fun = pyaldata.generate_epoch_fun(
        start_point_name='idx_cstStartTime',
        end_point_name='idx_cstEndTime',
        )

td_cst_hold = pyaldata.restrict_to_interval(
        td_cst,
        epoch_fun=hold_slice_fun,
        reset_index=False,
        )
td_cst_move = pyaldata.restrict_to_interval(
        td_cst,
        epoch_fun=move_slice_fun,
        reset_index=False,
        )

sig = 'hand_speed'
max_speed = np.percentile(abs(np.concatenate(td_cst_move['hand_speed'].values)),99,axis=0)
hold_thresh = np.percentile(abs(np.concatenate(td_cst_hold['hand_speed'].values)),80,axis=0)
speed_fig = plt.figure(figsize=(10,5))
# pos_fig,pos_ax = plt.subplots(figsize=(10,5))
@interact(trial_id = td_cst.index)
def plot_hold_move_hist(trial_id):
    cst.submovements.plot_hold_move_speed(
        td_cst.loc[trial_id,:],
        fig=speed_fig,
        max_speed=max_speed,
        hold_thresh=hold_thresh,
        hold_slice_fun=hold_slice_fun,
        move_slice_fun=move_slice_fun
    )
#     cst.plot_cst_traces(
#         trialtime = td_cst_move.loc[trial_id,'bin_size']*np.arange(td_cst_move.loc[trial_id,'rel_hand_pos'].shape[0]),
#         cursor_pos = td_cst_move.loc[trial_id,'rel_cursor_pos'][:,0],
#         hand_pos = td_cst_move.loc[trial_id,'rel_hand_pos'][:,0],
#         ax = pos_ax,
#     )
