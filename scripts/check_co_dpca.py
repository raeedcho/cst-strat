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
    td = cst.load_clean_data(args.infile,args.verbose)

    params = yaml.safe_load(open("params.yaml"))['co_dpca']
    sns.set_context('talk')

    td['M1_rates'] = [
        pyaldata.smooth_data(
            spikes/bin_size,
            dt=bin_size,
            std=params['smoothing_kernel_std'],
            backend='convolve',
        ) for spikes,bin_size in zip(td['M1_spikes'],td['bin_size'])
    ]
    
    # preprocess M1 state
    td['M1_state'] = [rates.copy() for rates in td['M1_rates']]
    if params['softnorm_rates']:
        td = pyaldata.soft_normalize_signal(td,'M1_state')
    # zero mean signal across both tasks? Or just for CO before dPCA?

    td_co = td.groupby(['task','result']).get_group(('CO','R'))

    # Trim CO data to time around go cue
    epoch_fun = cst.generate_realtime_epoch_fun(
        'idx_goCueTime',
        rel_start_time=params['rel_start_time'],
        rel_end_time=params['rel_end_time'],
    )
    td_co = pyaldata.restrict_to_interval(td_co,epoch_fun=epoch_fun)

    # center signals for just CO data
    td_co = pyaldata.center_signal(td_co,'M1_state')

    td_co,latent_dict = cst.run_dpca(td_co)

    # temp plotting
    plot_dpca(td_co,latent_dict,params)

@pyaldata.copy_td
def run_dpca(td):
    '''
    Runs dPCA analysis on the trial data
    '''

    # Compose the neural data tensor
    # This is a 4D tensor with dimensions (num_trials, num_neurons, num_time_bins, num_targets)
    neural_tensor = cst.form_neural_tensor(td,'M1_state',cond_cols=['tgtDir','tgtMag'])

    # set up dpca
    dpca = dPCA(labels='st',join={'s':['s','st']},regularizer='auto')
    dpca.protect = ['t']

    # fit dPCA
    latent_dict = dpca.fit_transform(np.mean(neural_tensor,axis=0),trialX=neural_tensor)

    # add dPC projections to trial data
    for key,cond in {'t':'time','s':'target'}.items():
        td['M1_dpca_'+cond] = [dpca.transform(rates.T,marginalization=key).T for rates in td['M1_state']]

    return td,latent_dict

def plot_dpca(td,latent_dict,params):
    # TODO: fix this to only rely on the trial data
    timevec = np.arange(td['M1_state'].values[0].shape[0])*td['bin_size'].values[0]+params['rel_start_time']
    fig,ax = plt.subplots(2,5,figsize=(15,7),sharex=True,sharey=True)

    for condnum,(key,val) in enumerate(latent_dict.items()):
        for component in range(5):
            for s in range(4):
                ax[condnum,component].plot(timevec,val[component,s,:])

    ax[-1,0].set_xlabel('Time from go cue (s)')
    ax[0,0].set_ylabel('Time-related (Hz)')
    ax[1,0].set_ylabel('Target-related (Hz)')
    for component in range(5):
        ax[0,component].set_title(f'Component #{component+1}')

    sns.despine(fig=fig,trim=True)

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Examine dPCA components of Center-Out task')
    parser.add_argument('--infile',type=str,help='Path to trial_data .mat file')
    parser.add_argument('--outdir',type=str,help='Path to output directory for results and figures')
    parser.add_argument('-v','--verbose',action='store_true',help='Verbose output')
    args = parser.parse_args()
    main(args)