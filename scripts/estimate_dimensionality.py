#!/bin/python3
# This script uses factor analysis cross-validated log-likelihood to estimate the dimensionality
# of the neural population in each task and epoch.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import src
import pyaldata
import os
import yaml

from sklearn.decomposition import FactorAnalysis
from sklearn.model_selection import RepeatedKFold

def main(args):
    td = src.load_clean_data(args.infile,args.verbose)

    with open('params.yaml') as params_file:
        params = yaml.safe_load(params_file)['dimensionality']

    td['M1_rates'] = [
        pyaldata.smooth_data(
            spikes/bin_size,
            dt=bin_size,
            std=params['smoothing_kernel_std'],
            backend='convolve',
        ) for spikes,bin_size in zip(td['M1_spikes'],td['bin_size'])
    ]

    td_epochs = src.split_hold_move(td)
    td_epochs = pyaldata.combine_time_bins(td_epochs,int(params['analysis_bin_size']//td_epochs['bin_size'].values[0]))

    # get only successful trials
    td_epochs = td_epochs.groupby('result').get_group('R')

    # split, aggregate, and concatenate...
    num_dims_list = np.arange(1,td_epochs['M1_rates'].values[0].shape[1]+1)
    cvLL_task_epoch = (
        td_epochs.groupby(['task','epoch'])['M1_rates']
        .apply(lambda x: sweep_dims_for_cvll(x,num_dims_list,params['n_folds'],params['n_repeats']))
    )
    # save this to a feather file...
    # savename = src.format_outfile_name(td,postfix='cvLL_task_epoch')
    # cvLL_task_epoch.reset_index().to_feather(os.path.join(args.outdir,savename+'.feather'))

    # temp figure (to be functionalized later)
    sns.lineplot(data=cvLL_task_epoch,x='num_dims',y='log_likelihood',hue='task',style='epoch')
    sns.despine(trim=True)

    sns.set_context('talk')
    fig_gen_dict = {
    }

    if not os.path.isdir(args.outdir):
        os.mkdir(args.outdir)

    for fig_postfix,fig in fig_gen_dict.items():
        fig_name = src.format_outfile_name(td,postfix=fig_postfix)
        fig.savefig(os.path.join(args.outdir,fig_name+'.png'))
        # fig.savefig(os.path.join(args.outdir,fig_name+'.pdf'))

def score_fa_fold(train_data,test_data,num_dims):
    '''
    Compute the cross-validated log-likelihood of factor analysis for a given number of dimensions

    Arguments:
        train_data: (np.array) data for training
        test_data: (np.array) data for testing
        num_dims: (int) number of dimensions to use for factor analysis

    Returns:
        float: cross-validated log-likelihood of factor analysis
    '''
    fa_model = FactorAnalysis(n_components=num_dims)
    fa_model.fit(train_data)
    return fa_model.score(test_data)

def sweep_dims_for_cvll(trial_firing_rates,num_dim_list,n_folds=5,n_repeats=10):
    '''
    For a group of trials in td, iterate through number of dimensions in num_dim_list
    and compute cross-validated log-likelihood of factor analysis for each number of dimensions.
    Note: crossval fold splits are the same for each element of num_dim_list,
    allowing for grouped comparisons.

    Arguments:
        trial_firing_rates: (pd.Series of np.array) column out of trial_data of firing rate arrays for each trial
        num_dim_list: (list) list of numbers of dimensions to use for factor analysis
        n_folds: (int) number of folds to use for cross-validation
        n_repeats: (int) number of times to repeat cross-validation

    Returns:
        pandas.DataFrame: dataframe with rows corresponding to one cv-fold of log likelihood for
            each element of num_dim_list. Index for each row will be:
            - (num_dims,fold_index)
    '''

    rkf = RepeatedKFold(n_splits=n_folds,n_repeats=n_repeats)
    cvLL_list = []
    for fold_idx, (train_idx, test_idx) in enumerate(rkf.split(trial_firing_rates)):
        for num_dims in num_dim_list:
            train_data = np.row_stack(trial_firing_rates.values[train_idx])
            test_data = np.row_stack(trial_firing_rates.values[test_idx])
            LL = score_fa_fold(train_data,test_data,num_dims)
            cvLL_list.append({'num_dims':num_dims,'fold_index':fold_idx,'log_likelihood':LL})

    return pd.DataFrame(cvLL_list).set_index(['num_dims','fold_index'])