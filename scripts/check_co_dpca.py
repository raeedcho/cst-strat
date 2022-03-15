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

    td = preprocess_neural_activity(td,params)

    # Trim CO data to time around go cue
    td_co = td.groupby(['task','result']).get_group(('CO','R'))
    epoch_fun = cst.generate_realtime_epoch_fun(
        'idx_goCueTime',
        rel_start_time=params['rel_start_time'],
        rel_end_time=params['rel_end_time'],
    )
    td_co = pyaldata.restrict_to_interval(td_co,epoch_fun=epoch_fun)

    # center signals for just CO data for dPCA
    td_co = pyaldata.center_signal(td_co,'M1_state')

    dpca,latent_dict = fit_dpca(td_co)

    # add dPCA projections to trial data
    td_co = add_dpca_projections(td_co,dpca)

    # get CST data
    td_cst = td.groupby(['task','result']).get_group(('CST','R'))
    cst_epoch_fun = cst.generate_realtime_epoch_fun(
        'idx_cstStartTime',
        rel_start_time=0,
        rel_end_time=5,
    )
    td_cst = pyaldata.restrict_to_interval(td_cst,epoch_fun=cst_epoch_fun)

    # add dpca projections to CST data
    td_cst = add_dpca_projections(td_cst,dpca)

    # temp plotting
    fig_gen_dict = {
        'mean_co_dpca':plot_dpca(td_co,latent_dict,params),
        'co_dpca_trials': plot_dpca_projection(td_co,params),
        'cst_dpca_trials': plot_dpca_projection(td_cst,params),
    }

    for fig_postfix,fig in fig_gen_dict.items():
        fig_name = cst.format_outfile_name(td,postfix=fig_postfix)
        fig.savefig(os.path.join(args.outdir,fig_name+'.png'))
        # fig.savefig(os.path.join(args.outdir,fig_name+'.pdf'))


def preprocess_neural_activity(td,params):
    '''
    Preprocess neural activity for dPCA:
    - Adds firing rates through smoothing
    - Soft normalizes the signal if requested
    - Centers the neural state across all recorded times
    '''
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
    td = pyaldata.center_signal(td,'M1_state')

    return td

@pyaldata.copy_td
def fit_dpca(td):
    '''
    Runs dPCA analysis on the trial data

    Arguments:
        td (pandas.DataFrame): Trial data in pyaldata format

    Returns:
        (DPCA): dPCA model object
        (dict): Dictionary of latent variables for each marginalization
    '''

    # Compose the neural data tensor
    # This is a 4D tensor with dimensions (num_trials, num_neurons, num_targets, num_time_bins)
    neural_tensor = cst.form_neural_tensor(td,'M1_state',cond_cols='tgtDir')

    # set up dpca
    dpca = dPCA(labels='st',join={'s':['s','st']},regularizer='auto')
    dpca.protect = ['t']

    # fit dPCA
    latent_dict = dpca.fit_transform(np.mean(neural_tensor,axis=0),trialX=neural_tensor)

    return dpca,latent_dict

def add_dpca_projections(td,dpca):
    # add dPC projections to trial data
    for key,cond in {'t':'time','s':'target'}.items():
        td['M1_dpca_'+cond] = [dpca.transform(rates.T,marginalization=key).T for rates in td['M1_state']]

    return td

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

    return fig

def plot_dpca_projection(td,params):
    '''
    Plot out dPCA projections from td
    '''
    timevec = np.arange(td['M1_state'].values[0].shape[0])*td['bin_size'].values[0]+params['rel_start_time']
    fig,ax = plt.subplots(2,5,figsize=(15,7),sharex=True,sharey=True)

    # tgt_colors = {0: '#1f77b4', 90: '#ff7f0e', 180: '#2ca02c', -90: '#d62728'}
    for condnum,val in enumerate(['time','target']):
        # plot top five components of each marginalization
        for component in range(5):
            # # group by target direction
            # for tgtDir,td_temp in td.groupby('tgtDir'):
            # plot each trial
            for _,trial in td.iterrows():
                ax[condnum,component].plot(
                    timevec,
                    trial['M1_dpca_'+val][:,component],
                    color='k',
                )

    # pretty up
    ax[-1,0].set_xlabel('Time from go cue (s)')
    ax[0,0].set_ylabel('Time-related (Hz)')
    ax[1,0].set_ylabel('Target-related (Hz)')
    for component in range(5):
        ax[0,component].set_title(f'Component #{component+1}')
    sns.despine(fig=fig,trim=True)

    return fig

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Examine dPCA components of Center-Out task')
    parser.add_argument('--infile',type=str,help='Path to trial_data .mat file')
    parser.add_argument('--outdir',type=str,help='Path to output directory for results and figures')
    parser.add_argument('-v','--verbose',action='store_true',help='Verbose output')
    args = parser.parse_args()
    main(args)