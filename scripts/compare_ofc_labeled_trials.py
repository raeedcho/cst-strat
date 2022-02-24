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
    td_cst = extract_cst_epoch(labeled_td)
    td_epochs = extract_td_epochs(labeled_td)
    td_train_move,td_test_move = cst.apply_models(td_epochs,train_epochs=['move'],test_epochs=['full'],label_col='ControlPolicy')
    td_train_hold,td_test_hold = cst.apply_models(td_epochs,train_epochs=['hold'],test_epochs=['full'],label_col='ControlPolicy')

    sns.set_context('talk')

    fig_gen_dict = {
        'ofc_trial_label_avg_fr_scatter': plot_single_neuron_stats(td_cst),
        'ofc_trial_label_pca_traces': plot_pca_traces(td_cst),
        # Move-trained models
        'ofc_label_move_M1_pca': cst.plot_M1_hold_pca(td_train_move,label_col='ControlPolicy',hue_order=['Position','Velocity']),
        'ofc_label_move_M1_lda': cst.plot_M1_lda(td_train_move,label_col='ControlPolicy',hue_order=['Position','Velocity']),
        'ofc_label_move_beh': cst.plot_hold_behavior(td_train_move,label_col='ControlPolicy',hue_order=['Position','Velocity']),
        'ofc_label_move_beh_lda': cst.plot_beh_lda(td_train_move,label_col='ControlPolicy',hue_order=['Position','Velocity']),
        # Hold-trained models
        'ofc_label_hold_M1_pca': cst.plot_M1_hold_pca(td_train_hold,label_col='ControlPolicy',hue_order=['Position','Velocity']),
        'ofc_label_hold_M1_lda': cst.plot_M1_lda(td_train_hold,label_col='ControlPolicy',hue_order=['Position','Velocity']),
        'ofc_label_hold_beh': cst.plot_hold_behavior(td_train_hold,label_col='ControlPolicy',hue_order=['Position','Velocity']),
        'ofc_label_hold_beh_lda': cst.plot_beh_lda(td_train_hold,label_col='ControlPolicy',hue_order=['Position','Velocity']),
        # LDA traces
        'ofc_label_move_lda_trace': cst.plot_M1_lda_traces(td_test_move,label_col='ControlPolicy',label_colors={'Position':'r','Velocity':'b'}),
        'ofc_label_hold_lda_trace': cst.plot_M1_lda_traces(td_test_hold,label_col='ControlPolicy',label_colors={'Position':'r','Velocity':'b'}),
        # 'task_lda_trace_avg':plot_M1_lda_traces(td_test_move_avg)
    }

    for fig_postfix,fig in fig_gen_dict.items():
        fig_name = cst.format_outfile_name(labeled_td,postfix=fig_postfix)
        fig.savefig(os.path.join(args.outdir,fig_name+'.png'))
        # fig.savefig(os.path.join(args.outdir,fig_name+'.pdf'))

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

    # TODO: generalize this to load an arbitrary label file
    temp = scipy.io.loadmat('data/ofc_trial_label_struct.mat',simplify_cells=True)
    labels = pd.DataFrame(temp['ofc_trial_labels'])
    labels['session_date'] = pd.to_datetime(labels['date'])

    labels = labels.loc[
        (labels['ControlPolicy']=='Velocity') | (labels['ControlPolicy']=='Position'),
        ['monkey','session_date','trial_id','ControlPolicy']]

    labeled_td = td.merge(labels)

    return labeled_td

@pyaldata.copy_td
def extract_cst_epoch(td):
    '''
    Pull out epochs of interest from trial data to check for differences between OFC labels
    '''
    td['M1_rates'] = [
        pyaldata.smooth_data(
            spikes/bin_size,
            dt=bin_size,
            std=0.05,
            backend='convolve',
        ) for spikes,bin_size in zip(td['M1_spikes'],td['bin_size'])
    ]
    td = pyaldata.restrict_to_interval(
        td,
        start_point_name='idx_cstStartTime',
        end_point_name='idx_cstEndTime',
        reset_index=False
    )
    td = pyaldata.combine_time_bins(td,int(0.05/td['bin_size'].values[0]))

    pca_model = PCA(n_components=8)
    td = pyaldata.dim_reduce(td,pca_model,'M1_rates','M1_pca')

    return td

@pyaldata.copy_td
def extract_td_epochs(td):
    '''
    Pull out epochs of interest from trial data to check for differences between OFC labels
    '''
    binned_epoch_dict = {
        'hold': cst.generate_realtime_epoch_fun(
            'idx_goCueTime',
            rel_start_time = -0.4,
            rel_end_time = 0,
        ),
        'move': cst.generate_realtime_epoch_fun(
            'idx_cstStartTime',
            rel_start_time = 0,
            rel_end_time = 5,
        ),
    }
    td_binned = cst.split_trials_by_epoch(td,binned_epoch_dict)
    # TODO: make the following lines more extensible by creating a "combine_all_time_bins" function
    td_binned = pd.concat([
        pyaldata.combine_time_bins(
            td_binned.loc[td_binned['epoch']=='hold',:],
            int(0.4/td_binned['bin_size'].values[0])
        ),
        pyaldata.combine_time_bins(
            td_binned.loc[td_binned['epoch']=='move',:],
            int(5/td_binned['bin_size'].values[0])
        )
    ]).reset_index(drop=True)
    assert td_binned['M1_spikes'].values[0].ndim==1, "Binning didn't work"
    td_binned['M1_rates'] = [spikes/bin_size for spikes,bin_size in zip(td_binned['M1_spikes'],td_binned['bin_size'])]

    smooth_epoch_dict = {
        'hold_move': pyaldata.generate_epoch_fun(
            'idx_goCueTime',
            rel_start=-0.4/td['bin_size'].values[0],
            rel_end=0.5/td['bin_size'].values[0],
        ),
        'full': pyaldata.generate_epoch_fun(
            'idx_goCueTime',
            end_point_name='idx_endTime',
            rel_start=-0.4/td['bin_size'].values[0],
            rel_end=-1,
        )
    }
    td_smooth = td.copy()
    td_smooth['M1_rates'] = [
        pyaldata.smooth_data(
            spikes,
            dt=bin_size,
            std=0.05,
            backend='convolve',
        ) for spikes,bin_size in zip(td_smooth['M1_spikes'],td_smooth['bin_size'])
    ]
    td_smooth = cst.split_trials_by_epoch(td_smooth,smooth_epoch_dict)
    td_smooth = pyaldata.combine_time_bins(td_smooth,int(0.05/td_smooth['bin_size'].values[0]))

    td_epochs = pd.concat([td_binned,td_smooth]).reset_index(drop=True)

    return td_epochs

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