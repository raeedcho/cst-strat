#!/bin/python3
# This script simply plots out the neural traces for CO and CST for each neuron

import src
import pyaldata
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

def main(args):
    td = src.load_clean_data(args.infile,args.verbose)
    td = pyaldata.restrict_to_interval(
        td,
        start_point_name='idx_goCueTime',
        rel_start=-0.4//td['bin_size'].values[0],
        end_point_name='idx_endTime',
        rel_end=-1,
        reset_index=False
    )
    td = src.add_trial_time(td,ref_event='idx_goCueTime')

    sns.set_context('talk')
    for _,trial in td.iterrows():
        fig,[beh_ax,raster_ax] = plt.subplots(2,1,figsize=(10,8),sharex=True)
        src.plot_horizontal_hand_movement(trial,ax=beh_ax,events=['idx_goCueTime'],ref_event_idx=trial['idx_goCueTime'])
        src.make_trial_raster(trial,ax=raster_ax,events=['idx_goCueTime'],ref_event_idx=trial['idx_goCueTime'])

        beh_ax.set_xlabel('')
        beh_ax.set_title('')
        raster_ax.set_title('')

        fig_outfile_name = '{monkey}_{session_date}_trial{trial_id}_raster.png'.format(
            monkey=trial['monkey'],
            session_date=trial['session_date'].strftime('%Y%m%d'),
            trial_id=trial.name
        )
        plt.savefig(os.path.join(args.outdir,fig_outfile_name))
        plt.close(fig)

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run single neuron analyses')
    parser.add_argument('--infile',required=True,help='path to input file')
    parser.add_argument('--outdir',required=True,help='path to output directory for results and plots')
    parser.add_argument('-v', '--verbose', action='store_true', help='increase output verbosity')
    args = parser.parse_args()

    main(args)