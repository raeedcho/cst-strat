#!/bin/python3
# This script examines the consistency of the neural population covariance (manifold) across tasks and epochs

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cst
import pyaldata
import os
import yaml

def main(args):
    params = yaml.safe_load(open("params.yaml"))['subspace_consistency']
    sns.set_context('talk')

    td = cst.load_clean_data(args.infile,args.verbose)
    td['M1_rates'] = [
        pyaldata.smooth_data(
            spikes/bin_size,
            dt=bin_size,
            std=params['smoothing_kernel_std'],
            backend='convolve',
        ) for spikes,bin_size in zip(td['M1_spikes'],td['bin_size'])
    ]

    epoch_dict = {
        'hold': cst.generate_realtime_epoch_fun(
            'idx_goCueTime',
            rel_start_time=-0.4,
            rel_end_time=0,
        ),
        'move': cst.generate_realtime_epoch_fun(
            'idx_goCueTime',
            end_point_name='idx_endTime',
            rel_start_time=0,
            rel_end_time=0,
        ),
    }
    td_epochs = cst.split_trials_by_epoch(td,epoch_dict)
    td_epochs = pyaldata.combine_time_bins(td_epochs,int(params['analysis_bin_size']//td_epochs['bin_size'].values[0]))

    td_boots = bootstrap_subspace_overlap(td_epochs,params['num_bootstraps'])

    fig_gen_dict = {
        'task_epoch_subpsace_overlap': plot_subspace_overlap(td_boots),
    }

    for fig_postfix,fig in fig_gen_dict.items():
        fig_name = cst.format_outfile_name(td,postfix=fig_postfix)
        fig.savefig(os.path.join(args.outdir,fig_name+'.png'))
        # fig.savefig(os.path.join(args.outdir,fig_name+'.pdf'))

def bootstrap_subspace_overlap(td_epochs,num_bootstraps):
    '''
    Compute subspace overlap for each pair of tasks and epochs,
    with bootstrapping to get distributions

    Arguments:
        td_epochs: (pandas.DataFrame) trial data with epochs split by task and epoch
        num_bootstraps: (int) number of bootstraps to perform

    Returns:
        pandas.DataFrame: dataframe with rows corresponding to each bootstrap
            of subspace overlap computed for pairs of task/epoch combos
    '''
    td_epoch_grouped = td_epochs.groupby(['task','epoch'],as_index=False)
    td_boots = []
    for boot_id in range(num_bootstraps):
        data_td = td_epoch_grouped.agg(
            M1_rates = ('M1_rates',lambda rates : np.row_stack(rates.sample(frac=1,replace=True)))
        )
        proj_td = td_epoch_grouped.agg(
            M1_rates = ('M1_rates',lambda rates : np.row_stack(rates.sample(frac=1,replace=True)))
        )
        td_pairs = data_td.join(
            proj_td,
            how='cross',
            lsuffix='_data',
            rsuffix='_proj',
        )

        td_pairs['boot_id'] = boot_id
        td_boots.append(td_pairs)
    
    td_boots = pd.concat(td_boots).reset_index(drop=True)
    
    td_boots['subspace_overlap'] = [
        cst.subspace_overlap_index(data,proj,num_dims=10)
        for data,proj in zip(td_boots['M1_rates_data'],td_boots['M1_rates_proj'])
    ]

    td_boots['subspace_overlap_rand'] = [
        cst.subspace_overlap_index(data,cst.random_array_like(data),num_dims=10)
        for data in td_boots['M1_rates_data']
    ]

    return td_boots


def plot_subspace_overlap(td_boots):
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


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Compare neural manifold across tasks and epochs')
    parser.add_argument('--infile',type=str,help='Path to trial_data .mat file')
    parser.add_argument('--outdir',type=str,help='Path to output directory for results and figures')
    parser.add_argument('-v','--verbose',action='store_true',help='Verbose output')
    args = parser.parse_args()
    main(args)