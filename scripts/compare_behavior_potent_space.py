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

from sklearn.linear_model import LinearRegression

def main(args):
    td = src.load_clean_data(args.infile,args.verbose)

    with open('params.yaml') as params_file:
        params = yaml.safe_load(params_file)['potent_space']

    td['M1_rates'] = [
        pyaldata.smooth_data(
            spikes/bin_size,
            dt=bin_size,
            std=params['smoothing_kernel_std'],
            backend='convolve',
        ) for spikes,bin_size in zip(td['M1_spikes'],td['bin_size'])
    ]

    # get behavioral potent space to hand velocity for each task
    td_co = td.groupby(['task','result']).get_group(('CO','R'))
    td_cst = td.groupby(['task','result']).get_group(('CST','R'))

    # check subspace overlap between potent spaces of tasks
    # (use potent spaces to random signals with same frequency characteristics--maybe through TME?)

    sns.set_context('talk')
    fig_gen_dict = {
    }

    if not os.path.isdir(args.outdir):
        os.mkdir(args.outdir)

    for fig_postfix,fig in fig_gen_dict.items():
        fig_name = src.format_outfile_name(td,postfix=fig_postfix)
        fig.savefig(os.path.join(args.outdir,fig_name+'.png'))
        # fig.savefig(os.path.join(args.outdir,fig_name+'.pdf'))

def find_behavioral_potent_space(neural_sig,behav_sig):
    '''
    Find the behavioral potent space between neural state and hand velocity
    by orthonormalizing the coefficients of the linear regression of the
    neural state onto the hand velocity.

    Arguments:
        neural_sig: (np.ndarray) neural state (num_timepoints x num_neural_dims)
        behav_sig: (np.ndarray) hand velocity (num_timepoints x 3)

    Returns:
        (np.ndarray) behavioral potent space (num_timepoints x num_neural_dims)
    '''
    model = LinearRegression()
    model.fit(neural_sig,behav_sig)
    return src.orth_combine_subspaces(model._coef)

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Compare neural manifold across tasks and epochs')
    parser.add_argument('--infile',type=str,help='Path to trial_data .mat file')
    parser.add_argument('--outdir',type=str,help='Path to output directory for results and figures')
    parser.add_argument('-v','--verbose',action='store_true',help='Verbose output')
    args = parser.parse_args()
    main(args)
