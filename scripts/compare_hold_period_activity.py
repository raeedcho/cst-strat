#!/bin/python3
# This script compares hold period neural population activity between CO and CST trials

import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import cst
import pyaldata
import os

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA

def main(args):
    sns.set_context('talk')

    td = cst.load_clean_data(args.infile,args.verbose)
    td_epoch = extract_td_epochs(td)
    td_train,td_test = apply_models(td_epoch,train_epochs=['hold'],test_epochs=['hold_move'])
    # td_test_avg = pyaldata.trial_average(td_test,'task')

    fig_gen_dict = {
        'task_M1_pca':plot_M1_hold_pca(td_train),
        'task_M1_lda':plot_M1_lda(td_train),
        'task_beh':plot_hold_behavior(td_train),
        'task_beh_lda':plot_beh_lda(td_train),
        # LDA traces
        'task_lda_trace':plot_M1_lda_traces(td_test),
        # 'task_lda_trace_avg':plot_M1_lda_traces(td_test_avg)
    }

    for fig_postfix,fig in fig_gen_dict.items():
        fig_name = cst.format_outfile_name(td_hold,postfix=fig_postfix)
        fig.savefig(os.path.join(args.outdir,fig_name+'.png'))
        # fig.savefig(os.path.join(args.outdir,fig_name+'.pdf'))

def extract_td_epochs(td):
    '''
    Prepare data for hold-time PCA and LDA, as well as data for smooth hold/move M1 activity
    
    Arguments:
        args (Namespace): Namespace of command-line arguments
        
    Returns:
        td_binned (DataFrame): PyalData formatted structure of neural/behavioral data
        td_smooth (DataFrame): PyalData formatted structure of neural/behavioral data
    '''
    binned_epoch_dict = {
        'hold': pyaldata.generate_epoch_fun(
            'idx_goCueTime',
            rel_start=-0.4/td['bin_size'].values[0],
            rel_end=-1,
        ),
        'move': pyaldata.generate_epoch_fun(
            'idx_goCueTime',
            rel_start=0,
            rel_end=0.4/td['bin_size'].values[0],
        ),
    }
    td_binned = cst.split_trials_by_epoch(td,binned_epoch_dict)
    td_binned = pyaldata.combine_time_bins(td_binned,int(0.4/td_binned['bin_size'].values[0]))
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

    td_epochs = pd.concat([td_binned,td_smooth]).reset_index()

    return td_epochs

@pyaldata.copy_td
def apply_models(td,train_epochs=['hold'],test_epochs=['hold_move']):
    '''
    Apply PCA and LDA models to hold-time data

    Note: only returns the data of the chosen epochs
    '''
    
    if type(train_epochs)==str:
        train_epochs = [train_epochs]
    if type(test_epochs)==str:
        test_epochs = [test_epochs]

    assert type(train_epochs)==list, "train_epochs must be a list"
    assert type(test_epochs)==list, "test_epochs must be a list"

    td_train = td.loc[td['epoch'].isin(train_epochs),:].copy()
    td_test = td.loc[td['epoch'].isin(test_epochs),:].copy()
    pca_model = PCA(n_components=8)
    pca_model.fit(np.row_stack(td_train['M1_rates']))
    td_train['M1_pca'] = [pca_model.transform(rates) for rates in td_train['M1_rates']]
    td_test['M1_pca'] = [pca_model.transform(rates) for rates in td_test['M1_rates']]

    M1_lda_model = LinearDiscriminantAnalysis()
    td_train['M1_lda'] = M1_lda_model.fit_transform(
        np.row_stack(td_train['M1_pca'].values),
        td_train['task']
    )
    td_train['M1_task_pred'] = M1_lda_model.predict(np.row_stack(td_train['M1_pca']))
    td_test['M1_lda'] = [M1_lda_model.transform(sig) for sig in td_test['M1_pca']]

    beh_lda_model = LinearDiscriminantAnalysis()
    td_train['beh_lda'] = beh_lda_model.fit_transform(
        np.column_stack([
            np.row_stack(td_train['rel_hand_pos'].values),
            np.row_stack(td_train['hand_vel'].values),
        ]),
        td_train['task']
    )
    td_train['beh_task_pred'] = beh_lda_model.predict(
        np.column_stack([
            np.row_stack(td_train['rel_hand_pos'].values),
            np.row_stack(td_train['hand_vel'].values),
        ])
    )

    return td_train,td_test

def plot_M1_hold_pca(td_hold):
    '''
    Plot the M1 neural population activity for CO and CST trials

    Arguments:
        td_hold (DataFrame): PyalData formatted structure of neural/behavioral data

    Returns:
        pca_fig (Figure): Figure of PCA plot
    '''
    pca_fig,pca_ax = plt.subplots(1,1,figsize=(8,6))
    sns.scatterplot(
        ax=pca_ax,
        data=td_hold,
        x=np.row_stack(td_hold['M1_pca'].values)[:,0],
        y=np.row_stack(td_hold['M1_pca'].values)[:,1],
        hue='task',palette='Set1',hue_order=['CO','CST']
    )
    pca_ax.set_ylabel('M1 PC2')
    pca_ax.set_xlabel('M1 PC1')
    sns.despine(ax=pca_ax,trim=True)

    return pca_fig

def plot_M1_lda_traces(td_smooth):
    '''
    Plot out M1 activity through hold period and first part of trial
    projected through LDA axis fit on average hold activity to separate
    tasks.

    Arguments:
        td_smooth (DataFrame): PyalData formatted structure of neural/behavioral data

    Returns:
        lda_fig (Figure): Figure of LDA plot
    '''
    task_colors = {'CO':'r','CST':'b'}
    lda_fig,lda_ax = plt.subplots(1,1,figsize=(8,6))
    for _,trial in td_smooth.iterrows():
        trialtime = (np.arange(trial['M1_rates'].shape[0])-trial['idx_goCueTime'])*trial['bin_size']
        lda_ax.plot(
            trialtime,
            trial['M1_lda'][:,0],
            c=task_colors[trial['task']],
            alpha=0.2,
        )
    lda_ax.set_ylabel('M1 LDA')
    lda_ax.set_xlabel('Time from go cue (s)')
    sns.despine(ax=lda_ax,trim=True)

    return lda_fig

def plot_hold_behavior(td_hold):
    '''
    Plot out hold time behavior (hand position and velocity)

    Arguments:
        td_hold (DataFrame): PyalData formatted structure of neural/behavioral data

    Returns:
        fig (Figure): Figure of behavior plot
    '''
    fig,[pos_ax,vel_ax] = plt.subplots(1,2,figsize=(10,6))
    sns.scatterplot(
        ax=pos_ax,
        data=td_hold,
        x=np.row_stack(td_hold['rel_hand_pos'].values)[:,0],
        y=np.row_stack(td_hold['rel_hand_pos'].values)[:,1],
        hue='task',palette='Set1',hue_order=['CO','CST']
    )
    pos_ax.set_aspect('equal')
    pos_ax.set_xlabel('X')
    pos_ax.set_ylabel('Y')
    pos_ax.set_title('Hand position (mm)')
    sns.scatterplot(
        ax=vel_ax,
        data=td_hold,
        x=np.row_stack(td_hold['hand_vel'].values)[:,0],
        y=np.row_stack(td_hold['hand_vel'].values)[:,1],
        hue='task',palette='Set1',hue_order=['CO','CST']
    )
    vel_ax.legend_.remove()
    vel_ax.set_aspect('equal')
    vel_ax.set_xlabel('X')
    vel_ax.set_title('Hand velocity (mm/s)')

    sns.despine(fig=fig,trim=True)

    return fig

def plot_M1_lda(td_hold):
    '''
    Plot the M1 neural population activity LDA for CO and CST trials

    Arguments:
        td_hold (DataFrame): PyalData formatted structure of neural/behavioral data

    Returns:
        lda_fig (Figure): Figure of lda plot

    TODO: add discriminability text somewhere in this plot
    '''
    # Hold-time LDA
    lda_fig,lda_ax = plt.subplots(1,1,figsize=(8,6))
    lda_ax.plot([0,td_hold.shape[0]],[0,0],'k--')
    sns.scatterplot(ax=lda_ax,data=td_hold,y='M1_lda',x='trial_id',hue='task',palette='Set1',hue_order=['CO','CST'])
    lda_ax.text(0,-3,'Discriminability: {:.2f}'.format((td_hold['M1_task_pred']==td_hold['task']).mean()))
    lda_ax.set_ylabel('M1 LDA projection')
    lda_ax.set_xlabel('Trial ID')
    sns.despine(ax=lda_ax,trim=True)

    return lda_fig

def plot_beh_lda(td_hold):
    '''
    Plot the behavioral LDA for CO and CST trials

    Arguments:
        td_hold (DataFrame): PyalData formatted structure of neural/behavioral data

    Returns:
        lda_fig (Figure): Figure of lda plot
    '''
    # Hold-time LDA
    lda_fig,lda_ax = plt.subplots(1,1,figsize=(8,6))
    lda_ax.plot([0,td_hold.shape[0]],[0,0],'k--')
    sns.scatterplot(ax=lda_ax,data=td_hold,y='beh_lda',x='trial_id',hue='task',palette='Set1',hue_order=['CO','CST'])
    lda_ax.text(0,-3,'Discriminability: {:.2f}'.format((td_hold['beh_task_pred']==td_hold['task']).mean()))
    lda_ax.set_ylabel('Behavioral LDA projection')
    lda_ax.set_xlabel('Trial ID')
    sns.despine(ax=lda_ax,trim=True)

    return lda_fig

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Compare hold period activity between CO and CST trials')
    parser.add_argument('--infile',required=True,help='path to input file')
    parser.add_argument('--outdir',required=True,help='path to output directory for results and plots')
    parser.add_argument('-v', '--verbose', action='store_true', help='increase output verbosity')
    args = parser.parse_args()

    main(args)