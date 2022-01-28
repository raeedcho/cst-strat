#!/bin/python3
# This script compares trials labeled as position- or velocity-controlled via OFC simulation

import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import cst
import pyaldata
import os
from sklearn.decomposition import PCA

def main(args):
    labeled_td = load_labeled_td(args.infile,args.verbose)

    sns.set_context('talk')

    single_fr_stats_fig = plot_single_neuron_stats(labeled_td)
    fig_outfile_name = cst.format_outfile_name(labeled_td,postfix='ofc_trial_label_avg_fr_scatter.png')
    single_fr_stats_fig.savefig(os.path.join(args.outdir,fig_outfile_name))

    pca_fig = plot_pca_traces(labeled_td)
    fig_outfile_name = cst.format_outfile_name(labeled_td,postfix='ofc_trial_label_pca_traces.png')
    pca_fig.savefig(os.path.join(args.outdir,fig_outfile_name))

def load_labeled_td(infile,verbose=False):
    '''
    Loads trial data file and OFC labels, then merges them together
    to return a labeled trial data DataFrame (only the active CST portion)
    Note: also adds a smoothed M1 rates signal to the trial_data
    
    Arguments:
        args (argparse.ArgumentParser): parsed command line arguments
        
    Returns:
        DataFrame: trial data with labels
    '''
    td = cst.load_clean_data(infile,verbose)
    td.reset_index(inplace=True)

    td['M1_rates'] = [
        pyaldata.smooth_data(spikes/bin_size,dt=bin_size,std=0.05) 
        for spikes,bin_size in zip(td['M1_spikes'],td['bin_size'])
    ]

    # TODO: generalize this to load an arbitrary label file
    temp = scipy.io.loadmat('data/ofc_trial_label_struct.mat',simplify_cells=True)
    labels = pd.DataFrame(temp['ofc_trial_labels'])
    labels['session_date'] = pd.to_datetime(labels['date'])

    labels = labels.loc[(labels['ControlPolicy']=='Velocity') | (labels['ControlPolicy']=='Position'),:]

    labeled_td = td.merge(labels)

    labeled_td = pyaldata.restrict_to_interval(
        labeled_td,
        start_point_name='idx_cstStartTime',
        end_point_name='idx_cstEndTime',
        reset_index=False
    )

    labeled_td = pyaldata.combine_time_bins(labeled_td,int(0.05/labeled_td['bin_size'].values[0]))

    pca_model = PCA(n_components=8)
    labeled_td = pyaldata.dim_reduce(labeled_td,pca_model,'M1_rates','M1_pca')

    return labeled_td

def plot_single_neuron_stats(labeled_td):
    '''
    Plots scatter of average firing rate for position or velocity controlled trials

    Arguments:
        labeled_td (DataFrame): trial data with labels

    Returns:
        fig (Figure): figure object
    '''
    # avg_fr_table = cst.get_condition_neural_averages(labeled_td,signal='M1_spikes',cond_col='ControlPolicy')
    # max_fr = avg_fr_table['average_rate'].max()
    # avg_fr_pivot = avg_fr_table.pivot(index=['chan_id','unit_id'],columns='ControlPolicy',values='average_rate')

    fr_stats = labeled_td.groupby('ControlPolicy').agg(
        mean_fr=('M1_rates',lambda x: np.row_stack(x).mean(axis=0)),
        std_fr=('M1_rates',lambda x: np.row_stack(x).std(axis=0)),
    )
    max_fr = np.concatenate(fr_stats['mean_fr']).max()
    
    fig,ax = plt.subplots(1,2,figsize=(10,6))
    ax[0].plot([0,max_fr+5],[0,max_fr+5],'k--')
    sns.scatterplot(
        ax=ax[0],
        x=fr_stats.loc['Position','mean_fr'],
        y=fr_stats.loc['Velocity','mean_fr'],
    )
    ax[0].set_xlabel('Position-Controlled trials')
    ax[0].set_ylabel('Velocity-Controlled trials')
    ax[0].set_title('Average Firing Rate (Hz)')

    ax[1].plot([0,np.sqrt(max_fr)+5],[0,np.sqrt(max_fr)+5],'k--')
    sns.scatterplot(
        ax=ax[1],
        x=fr_stats.loc['Position','std_fr'],
        y=fr_stats.loc['Velocity','std_fr'],
    )
    ax[1].set_xlabel('Position-Controlled trials')
    ax[1].set_ylabel('Velocity-Controlled trials')
    ax[1].set_title('Firing Rate Standard Deviation (Hz)')
    sns.despine(fig=fig,trim=True)

    return fig

def plot_pca_traces(labeled_td):
    '''
    Plots PCA traces of M1_rates from labeled_td, separated out by ControlPolicy
    
    Arguments:
        labeled_td (DataFrame): trial data with labels
        
    Returns:
        fig (Figure): figure object
    '''

    policy_colors = {
        'Position':'b',
        'Velocity':'r',
    }

    fig,ax = plt.subplots(1,1,figsize=(6,6))
    for policy,pol_td in labeled_td.groupby('ControlPolicy'):
        for _,trial in pol_td.iterrows():
            line, = ax.plot(trial['M1_pca'][:,0],trial['M1_pca'][:,1],policy_colors[policy])
        line.set_label(policy) 

    ax.set_xlabel('M1 PC1')
    ax.set_ylabel('M1 PC2')
    ax.legend()
    sns.despine(fig=fig,trim=True)

    return fig

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Compare trials labeled as position- or velocity-controlled via OFC simulation')
    parser.add_argument('--infile',required=True,help='Path to input trial_data file')
    parser.add_argument('--outdir',required=True,help='Path to output directory for results and plots')
    parser.add_argument('--verbose',action='store_true',help='Print verbose output')
    args = parser.parse_args()

    main(args)