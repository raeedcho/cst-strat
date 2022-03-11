# This script plots out the special dimensions of CO by using dPCA

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cst
import pyaldata
import os
import yaml
from dPCA.dPCA import dPCA

def main(args):
    params = yaml.safe_load(open("params.yaml"))['co_dpca']
    sns.set_context('talk')

    td = cst.load_clean_data(args.infile,args.verbose)

    # Get only CO data
    # add M1 rates to trial data
    # Trim CO data to time around go cue
    # add M1 state to trial data (softnormed?, zero mean)

    td = cst.run_dpca(td)

@pyaldata.copy_td
def run_dpca(td):
    '''
    Runs dPCA analysis on the trial data
    '''

    # Compose the neural data tensor
    # This is a 4D tensor with dimensions (num_trials, num_neurons, num_time_bins, num_targets)
    neural_tensor = cst.form_neural_tensor(td) # use groupby somehow

    # set up dpca
    dpca = dPCA(labels='ts',join={'s':['s','ts']},regularizer='auto')
    dpca.protect = ['t']

    # fit dPCA
    latent_dict = dpca.fit_transform(np.mean(neural_tensor,axis=0),trialX=neural_tensor)

    # add dPC projections to trial data
    for key,cond in {'t':'time','s':'target'}.items():
        td['M1_dpca_'+cond] = [dpca.transform(rates.T,marginalization=key).T for rates in td['M1_state']]

    return td

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Examine dPCA components of Center-Out task')
    parser.add_argument('--infile',type=str,help='Path to trial_data .mat file')
    parser.add_argument('--outdir',type=str,help='Path to output directory for results and figures')
    parser.add_argument('-v','--verbose',action='store_true',help='Verbose output')
    args = parser.parse_args()
    main(args)