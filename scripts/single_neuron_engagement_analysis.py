#!/bin/python3
# This script runs a single neuron engagement analysis, looking for
# how average neural activity changes as lambda changes

import src
import pyaldata

def main(args):
    td = src.load_clean_data(args.infile,args.verbose)

    td_cst = td.loc[td['task']=='CST',:].copy()
    td_cst = pyaldata.restrict_to_interval(
        td_cst,
        start_point_name='idx_cstStartTime',
        end_point_name='idx_cstEndTime',
        reset_index=False
    )
    td_cst = pyaldata.combine_time_bins()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run single neuron analyses')
    parser.add_argument('--infile',required=True,help='path to input file')
    parser.add_argument('--outdir',required=True,help='path to output directory for results and plots')
    parser.add_argument('-v', '--verbose', action='store_true', help='increase output verbosity')
    args = parser.parse_args()

    main(args)