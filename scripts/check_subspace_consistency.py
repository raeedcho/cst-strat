#!/bin/python3
# This script examines the consistency of the neural population covariance (manifold) across tasks and epochs

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import src
import pyaldata
import os
import yaml

def main(args):
    td = src.load_clean_data(args.infile,args.verbose)

    with open('params.yaml') as params_file:
        params = yaml.safe_load(params_file)['subspace_consistency']

    td['M1_rates'] = [
        pyaldata.smooth_data(
            spikes/bin_size,
            dt=bin_size,
            std=params['smoothing_kernel_std'],
            backend='convolve',
        ) for spikes,bin_size in zip(td['M1_spikes'],td['bin_size'])
    ]

    # check subspace consistency across tasks and epochs
    td_boots_task_epoch = find_task_epoch_overlap(td,params)

    # check subspace consistency across lambda
    td_boots_lambda = find_lambda_overlap(td,params)

    sns.set_context('talk')
    fig_gen_dict = {
        'task_epoch_subpsace_overlap': plot_task_epoch_overlap(td_boots_task_epoch),
        'lambda_subspace_overlap': plot_lambda_overlap(td_boots_lambda),
    }

    if not os.path.isdir(args.outdir):
        os.mkdir(args.outdir)

    for fig_postfix,fig in fig_gen_dict.items():
        fig_name = src.format_outfile_name(td,postfix=fig_postfix)
        fig.savefig(os.path.join(args.outdir,fig_name+'.png'))
        # fig.savefig(os.path.join(args.outdir,fig_name+'.pdf'))

def find_task_epoch_overlap(td,params):
    '''
    Find the (bootstrapped) subspace overlap between pairs of task/epoch combos

    Arguments:
        td: (pandas.DataFrame) trial data
        params: (dict) parameters for data preprocessing and bootstrapping

    Returns:
        pandas.DataFrame: dataframe with rows corresponding to each bootstrap
            of subspace overlap computed for pairs of task/epoch combos
    '''
    td_epochs = src.split_hold_move(td)
    td_epochs = pyaldata.combine_time_bins(td_epochs,int(params['analysis_bin_size']//td_epochs['bin_size'].values[0]))

    # subselect CO trials with only horizontal movement
    if params['subselect_co_horiz_trials']:
        is_vert_trial = (td_epochs['tgtDir']==180) | (td_epochs['tgtDir']==-90)
        td_epochs = td_epochs.loc[~is_vert_trial,:]

    td_grouped = td_epochs.groupby(['task','epoch'],as_index=False)
    td_boots = src.bootstrap_subspace_overlap(td_grouped,params['num_bootstraps'])
    
    return td_boots

def plot_task_epoch_overlap(td_boots):
    axs = td_boots.hist(
        column='subspace_overlap',
        by=['task_data','epoch_data','task_proj','epoch_proj'],
        figsize=(15,8),
        sharex=True,
        sharey=True,
        xrot=0,
        bins=np.linspace(0,1,40),
        color=[0.6,0.3,0.6]
    )

    # fix formatting and make pretty-ish
    for rownum,row in enumerate(axs):
        for colnum,ax in enumerate(row):
            ax.set_xlim([0,1])

            if rownum==3:
                ax.set_xlabel('Subspace overlap')
            
            oldtitle = ax.get_title().replace('(','').replace(')','').split(', ')
            newtitle = ' '.join(oldtitle[2:])
            newylabel = ' '.join(oldtitle[:2])
            
            if rownum==0:
                ax.set_title(newtitle)
            else:
                ax.set_title('')

            if colnum==0:
                ax.set_ylabel(newylabel)
            else:
                ax.set_ylabel('')

            # controls
            ax.hist(
                td_boots.loc[
                    (td_boots['task_data']==oldtitle[0]) &
                    (td_boots['epoch_data']==oldtitle[1]) &
                    (td_boots['task_proj']==oldtitle[2]) &
                    (td_boots['epoch_proj']==oldtitle[3]),
                    'subspace_overlap_rand'],
                bins=np.linspace(0,1,40),
                color=[0.8,0.8,0.8],
            )

    sns.despine()

    return plt.gcf()

def find_lambda_overlap(td,params):
    '''
    Find subspace overlap between pairs of lambda values for CST data

    Arguments:
        td: (pandas.DataFrame) trial data
        params: (dict) parameters for data preprocessing and bootstrapping

    Returns:
        pandas.DataFrame: dataframe with rows corresponding to each bootstrap
            of subspace overlap computed for pairs of lambda values
    '''
    # get only CST data
    td_cst = td.groupby('task').get_group('CST')
    td_cst = pyaldata.restrict_to_interval(
        td_cst,
        epoch_fun=src.generate_realtime_epoch_fun(
            'idx_goCueTime',
            end_point_name='idx_endTime',
        ),
    )
    td_cst = pyaldata.combine_time_bins(td_cst,int(params['analysis_bin_size']//td_cst['bin_size'].values[0]))

    td_grouped = td_cst.groupby('lambda',as_index=False)
    td_boots = src.bootstrap_subspace_overlap(td_grouped,params['num_bootstraps'])

    return td_boots

def plot_lambda_overlap(td_boots):
    axs = td_boots.hist(
        column='subspace_overlap',
        by=['lambda_data','lambda_proj'],
        figsize=(15,8),
        sharex=True,
        sharey=True,
        xrot=0,
        bins=np.linspace(0,1,40),
        color=[0.6,0.3,0.6]
    )

    # fix formatting and make pretty-ish
    for rownum,row in enumerate(axs):
        for colnum,ax in enumerate(row):
            ax.set_xlim([0,1])

            if rownum==axs.shape[0]-1:
                ax.set_xlabel('Subspace overlap')
            
            oldtitle = ax.get_title().replace('(','').replace(')','').split(', ')
            newtitle = oldtitle[1]
            newylabel = oldtitle[0]
            
            if rownum==0:
                ax.set_title(newtitle)
            else:
                ax.set_title('')

            if colnum==0:
                ax.set_ylabel(newylabel)
            else:
                ax.set_ylabel('')

            # controls
            ax.hist(
                td_boots.loc[
                    (td_boots['lambda_data']==float(oldtitle[0])) &
                    (td_boots['lambda_proj']==float(oldtitle[1])),
                    'subspace_overlap_rand'],
                bins=np.linspace(0,1,40),
                color=[0.8,0.8,0.8],
            )

    sns.despine()

    return plt.gcf()

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Compare neural manifold across tasks and epochs')
    parser.add_argument('--infile',type=str,help='Path to trial_data .mat file')
    parser.add_argument('--outdir',type=str,help='Path to output directory for results and figures')
    parser.add_argument('-v','--verbose',action='store_true',help='Verbose output')
    args = parser.parse_args()
    main(args)