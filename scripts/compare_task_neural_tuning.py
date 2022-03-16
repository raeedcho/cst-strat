#!/bin/python3
# This script runs a single neuron tuning comparison between center-out and CST

import src
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main(args):
    td = src.load_clean_data(args.infile,args.verbose)
    lr_selectivity_table = src.single_neuron_analysis.get_task_neural_lr_selectivity(td,signal='M1_spikes')
    
    outfile_name = '{monkey}_{session_date}_lr_selectivity.feather'.format(
        monkey=td['monkey'].values[0],
        session_date=np.datetime_as_string(td['session_date'].values[0],'D').replace('-','')
    )
    lr_selectivity_table.to_feather(os.path.join(args.outdir,outfile_name))

    sns.set_context('talk')
    src.plot_task_neural_lr_selectivity(lr_selectivity_table)
    fig_outfile_name = '{monkey}_{session_date}_task_lr_selectivity.png'.format(
        monkey=td['monkey'].values[0],
        session_date=np.datetime_as_string(td['session_date'].values[0],'D').replace('-','')
    )
    plt.savefig(os.path.join(args.outdir,fig_outfile_name))

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run single neuron analyses')
    parser.add_argument('--infile',required=True,help='path to input file')
    parser.add_argument('--outdir',required=True,help='path to output directory for results and plots')
    parser.add_argument('-v', '--verbose', action='store_true', help='increase output verbosity')
    args = parser.parse_args()

    main(args)