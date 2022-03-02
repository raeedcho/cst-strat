#!/bin/python3
# This script examines the consistency of the neural population covariance (manifold) across tasks and epochs

import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import cst
import pyaldata
import os

def main(args):
    # TODO: move parameters to JSON params file
    num_bootstraps = 100

    sns.set_context('talk')

    td = cst.load_clean_data(args.infile,args.verbose)
    td['M1_rates'] = [
        pyaldata.smooth_data(
            spikes/bin_size,
            dt=bin_size,
            std=0.05,
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
        'hold_move': cst.generate_realtime_epoch_fun(
            'idx_goCueTime',
            end_point_name='idx_endTime',
            rel_start_time=-0.4,
            rel_end_time=0,
        ),
        # 'full': cst.generate_realtime_epoch_fun(
        #     'idx_startTime',
        #     end_point_name='idx_endTime',
        #     rel_start_time=0,
        #     rel_end_time=0,
        # ),
    }
    td_epochs = cst.split_trials_by_epoch(td,epoch_dict)
    td_epochs = pyaldata.combine_time_bins(td_epochs,int(0.05//td_epochs['bin_size'].values[0]))

    td_epoch_grouped = td_epochs.groupby(['task','epoch'],as_index=False)
    td_boots = []
    for boot_id in range(num_bootstraps):
        left_td = td_epoch_grouped.agg(
            M1_rates = ('M1_rates',lambda rates : np.row_stack(rates.sample(frac=1,replace=True)))
        )
        right_td = td_epoch_grouped.agg(
            M1_rates = ('M1_rates',lambda rates : np.row_stack(rates.sample(frac=1,replace=True)))
        )
        td_pairs = left_td.join(
            right_td,
            how='cross',
            lsuffix='_left',
            rsuffix='_right',
        )

        td_pairs['boot_id'] = boot_id
        td_boots.append(td_pairs)
    
    td_boots = pd.concat(td_boots).reset_index(drop=True)
    
    td_boots['subspace_overlap'] = [
        cst.subspace_overlap_index(left,right,num_dims=10)
        for left,right in zip(td_boots['M1_rates_left'],td_boots['M1_rates_right'])
    ]

    axs = td_boots.hist(
        column='subspace_overlap',
        by=['task_left','epoch_left','task_right','epoch_right'],
        figsize=(20,20),
        bins=np.linspace(0,1,40),
    )
    for col in axs:
        for ax in col:
            ax.set_xlim([0,1])

    # group by left-right pairs of tasks and epochs, aggregate to get mean and CI of subspace overlap
    subspace_overlaps = td_boots.groupby(['task_left','epoch_left','task_right','epoch_right'],as_index=False).agg(
        subspace_overlap = ('subspace_overlap',np.mean),
        subspace_overlap_CIlo = ('subspace_overlap',lambda x : np.percentile(x,2.5)),
        subspace_overlap_CIhi = ('subspace_overlap',lambda x : np.percentile(x,97.5)),
    )

    overlap_matrix = td_pairs.pivot(
        index=['task_left','epoch_left'],
        columns=['task_right','epoch_right'],
        values='subspace_overlap'
    )

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Compare neural manifold across tasks and epochs')
    parser.add_argument('--infile',type=str,help='Path to trial_data .mat file')
    parser.add_argument('--outdir',type=str,help='Path to output directory for results and figures')
    parser.add_argument('-v','--verbose',action='store_true',help='Verbose output')
    args = parser.parse_args()
    main(args)