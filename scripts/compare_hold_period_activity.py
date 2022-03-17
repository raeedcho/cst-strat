#!/bin/python3
# This script compares hold period neural population activity between CO and CST trials

import pandas as pd
import seaborn as sns
import src
import pyaldata
import os
from pathlib import Path

def main(args):
    td = src.load_clean_data(args.infile,args.verbose)
    td_epoch = extract_td_epochs(td)

    # TODO: save this output to a feather file in outdir
    td_train,td_test = src.apply_models(td_epoch,train_epochs=['hold'],test_epochs=['hold_move'])

    sns.set_context('talk')
    fig_gen_dict = {
        'task_M1_pca':src.plot_M1_hold_pca(td_train),
        'task_M1_lda':src.plot_M1_lda(td_train),
        'task_beh':src.plot_hold_behavior(td_train),
        'task_beh_lda':src.plot_beh_lda(td_train),
        # LDA traces
        'task_lda_trace':src.plot_M1_lda_traces(td_test),
    }

    fig_path = args.outdir/'figs'
    fig_path.mkdir(parents=True, exist_ok=True)
    for fig_postfix,fig in fig_gen_dict.items():
        fig_name = src.format_outfile_name(td_train,postfix=fig_postfix)
        fig.savefig(fig_path/(fig_name+'.png'))
        # fig.savefig(os.path.join(args.outdir,fig_name+'.png'))
        # fig.savefig(os.path.join(args.outdir,fig_name+'.pdf'))

def extract_td_epochs(td):
    '''
    Prepare data for hold-time PCA and LDA, as well as data for smooth hold/move M1 activity
    
    Arguments:
        args (Namespace): Namespace of command-line arguments
        
    Returns:
        td_binned (DataFrame): PyalData formatted structure of neural/behavioral data
        td_smooth (DataFrame): PyalData formatted structure of neural/behavioral data
    '''
    binned_epoch_dict = {
        'hold': pyaldata.generate_epoch_fun(
            'idx_goCueTime',
            rel_start=-0.4/td['bin_size'].values[0],
            rel_end=-1,
        ),
        'move': pyaldata.generate_epoch_fun(
            'idx_goCueTime',
            rel_start=0,
            rel_end=0.4/td['bin_size'].values[0],
        ),
    }
    td_binned = src.split_trials_by_epoch(td,binned_epoch_dict)
    td_binned = pyaldata.combine_time_bins(td_binned,int(0.4/td_binned['bin_size'].values[0]))
    assert td_binned['M1_spikes'].values[0].ndim==1, "Binning didn't work"
    td_binned['M1_rates'] = [spikes/bin_size for spikes,bin_size in zip(td_binned['M1_spikes'],td_binned['bin_size'])]

    smooth_epoch_dict = {
        'hold_move': pyaldata.generate_epoch_fun(
            'idx_goCueTime',
            rel_start=-0.4/td['bin_size'].values[0],
            rel_end=0.5/td['bin_size'].values[0],
        ),
        'full': pyaldata.generate_epoch_fun(
            'idx_goCueTime',
            end_point_name='idx_endTime',
            rel_start=-0.4/td['bin_size'].values[0],
            rel_end=-1,
        )
    }
    td_smooth = td.copy()
    td_smooth['M1_rates'] = [
        pyaldata.smooth_data(
            spikes/bin_size,
            dt=bin_size,
            std=0.05,
            backend='convolve',
        ) for spikes,bin_size in zip(td_smooth['M1_spikes'],td_smooth['bin_size'])
    ]
    td_smooth = src.split_trials_by_epoch(td_smooth,smooth_epoch_dict)
    td_smooth = pyaldata.combine_time_bins(td_smooth,int(0.05/td_smooth['bin_size'].values[0]))

    td_epochs = pd.concat([td_binned,td_smooth]).reset_index()

    return td_epochs

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Compare hold period activity between CO and CST trials')
    parser.add_argument('--infile',required=True,help='path to input file')
    parser.add_argument('--outdir',type=Path,required=True,help='path to output directory for results and plots')
    parser.add_argument('-v', '--verbose', action='store_true', help='increase output verbosity')
    args = parser.parse_args()

    main(args)