#!/bin/python3
# This script runs a single neuron comparison between center-out and CST

import cst
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main(args):
    td = cst.load_clean_data(args.infile,args.verbose)
    avg_fr_table = cst.single_neuron_analysis.get_task_epoch_neural_averages(td)

    # outfile_name = '{monkey}_{session_date}_averageFR.feather'.format(
    #     monkey=td['monkey'].values[0],
    #     session_date=td['session_date'].values[0]
    # )
    # avg_fr_table.to_feather(os.path.join(args.outdir,outfile_name))

    # plot out the average firing rate comparison
    sns.set_context('talk') # TODO: make this a command line argument
    
    cst.single_neuron_analysis.plot_task_epoch_neural_averages(avg_fr_table)

    fig_outfile_name = '{monkey}_{session_date}_avg_fr_comparison.png'.format(
        monkey=td['monkey'].values[0],
        session_date=np.datetime_as_string(td['session_date'].values[0],'D').replace('-','')
    )
    plt.savefig(os.path.join(args.outdir,fig_outfile_name))

# %%
if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run single neuron analyses')
    parser.add_argument('--infile',required=True,help='path to input file')
    parser.add_argument('--outdir',required=True,help='path to output directory for results and plots')
    parser.add_argument('-v', '--verbose', action='store_true', help='increase output verbosity')
    args = parser.parse_args()

    main(args)