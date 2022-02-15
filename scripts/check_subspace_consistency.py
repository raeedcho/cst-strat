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
    sns.set_context('talk')

    td = cst.load_clean_data(args.infile,args.verbose)
    td['M1_rates'] = [
        pyaldata.smooth_data(
            spikes,
            dt=bin_size,
            std=0.05,
            backend='convolve',
        ) for spikes,bin_size in zip(td['M1_spikes'],td['bin_size'])
    ]

    epoch_dict = {
        'hold': pyaldata.generate_epoch_fun(
            'idx_goCueTime',
            rel_start=-0.4/td['bin_size'].values[0],
            rel_end=-1,
        ),
        'move': pyaldata.generate_epoch_fun(
            'idx_goCueTime',
            end_point_name='idx_endTime',
            rel_start=0,
            rel_end=-1,
        ),
        'hold_move': pyaldata.generate_epoch_fun(
            'idx_goCueTime',
            end_point_name='idx_endTime',
            rel_start=-0.4/td['bin_size'].values[0],
            rel_end=-1,
        ),
        'full': pyaldata.generate_epoch_fun(
            'idx_startTime',
            end_point_name='idx_endTime',
            rel_start=0,
            rel_end=-1,
        ),
    }
    td_epochs = cst.split_trials_by_epoch(td,epoch_dict)

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Compare neural manifold across tasks and epochs')
    parser.add_argument('--infile',type=str,help='Path to trial_data .mat file')
    parser.add_argument('--outdir',type=str,help='Path to output directory for results and figures')
    parser.add_argument('-v','--verbose',action='store_true',help='Verbose output')
    args = parser.parse_args()
    main(args)