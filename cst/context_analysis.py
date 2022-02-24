import pyaldata
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA

@pyaldata.copy_td
def apply_models(td,train_epochs=None,test_epochs=None,label_col='task'):
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
    td_train['M1_pca'] = list(pca_model.fit_transform(np.row_stack(td_train['M1_rates'])))
    # td_train['M1_pca'] = [pca_model.transform(rates) for rates in td_train['M1_rates']]
    td_test['M1_pca'] = [pca_model.transform(rates) for rates in td_test['M1_rates']]

    M1_lda_model = LinearDiscriminantAnalysis()
    td_train['M1_lda'] = M1_lda_model.fit_transform(
        np.row_stack(td_train['M1_pca'].values),
        td_train[label_col]
    )
    td_train['M1_pred'] = M1_lda_model.predict(np.row_stack(td_train['M1_pca']))
    td_test['M1_lda'] = [M1_lda_model.transform(sig) for sig in td_test['M1_pca']]

    beh_lda_model = LinearDiscriminantAnalysis()
    td_train['beh_lda'] = beh_lda_model.fit_transform(
        np.column_stack([
            np.row_stack(td_train['rel_hand_pos'].values),
            np.row_stack(td_train['hand_vel'].values),
        ]),
        td_train[label_col]
    )
    td_train['beh_pred'] = beh_lda_model.predict(
        np.column_stack([
            np.row_stack(td_train['rel_hand_pos'].values),
            np.row_stack(td_train['hand_vel'].values),
        ])
    )

    return td_train,td_test

def plot_M1_hold_pca(td,label_col='task',hue_order=['CO','CST']):
    '''
    Plot the M1 neural population activity for trials separated by label (e.g. task)

    Arguments:
        td (DataFrame): PyalData formatted structure of neural/behavioral data

    Returns:
        pca_fig (Figure): Figure of PCA plot
    '''
    assert type(label_col)==str, "label_col must be a string"

    pca_fig,pca_ax = plt.subplots(1,1,figsize=(8,6))
    sns.scatterplot(
        ax=pca_ax,
        data=td,
        x=np.row_stack(td['M1_pca'].values)[:,0],
        y=np.row_stack(td['M1_pca'].values)[:,1],
        hue=label_col,palette='Set1',hue_order=hue_order
    )
    pca_ax.set_ylabel('M1 PC2')
    pca_ax.set_xlabel('M1 PC1')
    sns.despine(ax=pca_ax,trim=True)

    return pca_fig

def plot_M1_lda_traces(td_smooth,label_col='task',label_colors={'CO':'r','CST':'b'}):
    '''
    Plot out M1 activity through hold period and first part of trial
    projected through LDA axis fit on average hold activity to separate
    tasks.

    Arguments:
        td_smooth (DataFrame): PyalData formatted structure of neural/behavioral data
        label_col (str): Column name of label (e.g. task)

    Returns:
        lda_fig (Figure): Figure of LDA plot
    '''
    assert type(label_col)==str, "label_col must be a string"

    lda_fig,lda_ax = plt.subplots(1,1,figsize=(8,6))
    for _,trial in td_smooth.iterrows():
        trialtime = (np.arange(trial['M1_rates'].shape[0])-trial['idx_goCueTime'])*trial['bin_size']
        lda_ax.plot(
            trialtime,
            trial['M1_lda'][:,0],
            c=label_colors[trial[label_col]],
            alpha=0.2,
        )
    lda_ax.set_ylabel('M1 LDA')
    lda_ax.set_xlabel('Time from go cue (s)')
    sns.despine(ax=lda_ax,trim=True)

    return lda_fig

def plot_hold_behavior(td_hold,label_col='task',hue_order=['CO','CST']):
    '''
    Plot out hold time behavior (hand position and velocity)

    Arguments:
        td_hold (DataFrame): PyalData formatted structure of neural/behavioral data

    Returns:
        fig (Figure): Figure of behavior plot
    '''
    assert type(label_col)==str, "label_col must be a string"

    fig,[pos_ax,vel_ax] = plt.subplots(1,2,figsize=(10,6))
    sns.scatterplot(
        ax=pos_ax,
        data=td_hold,
        x=np.row_stack(td_hold['rel_hand_pos'].values)[:,0],
        y=np.row_stack(td_hold['rel_hand_pos'].values)[:,1],
        hue=label_col,palette='Set1',hue_order=hue_order
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
        hue=label_col,palette='Set1',hue_order=hue_order
    )
    vel_ax.legend_.remove()
    vel_ax.set_aspect('equal')
    vel_ax.set_xlabel('X')
    vel_ax.set_title('Hand velocity (mm/s)')

    sns.despine(fig=fig,trim=True)

    return fig

def plot_M1_lda(td_hold,label_col='task',hue_order=['CO','CST']):
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
    lda_ax.plot([td_hold['trial_id'].min(),td_hold['trial_id'].max()],[0,0],'k--')
    sns.scatterplot(
        ax=lda_ax,
        data=td_hold,
        y='M1_lda',
        x='trial_id',
        hue=label_col,palette='Set1',hue_order=hue_order
    )
    lda_ax.text(0,-3,'Discriminability: {:.2f}'.format((td_hold['M1_pred']==td_hold[label_col]).mean()))
    lda_ax.set_ylabel('M1 LDA projection')
    lda_ax.set_xlabel('Trial ID')
    sns.despine(ax=lda_ax,trim=True)

    return lda_fig

def plot_beh_lda(td_hold,label_col='task',hue_order=['CO','CST']):
    '''
    Plot the behavioral LDA for CO and CST trials

    Arguments:
        td_hold (DataFrame): PyalData formatted structure of neural/behavioral data

    Returns:
        lda_fig (Figure): Figure of lda plot
    '''
    # Hold-time LDA
    lda_fig,lda_ax = plt.subplots(1,1,figsize=(8,6))
    lda_ax.plot([td_hold['trial_id'].min(),td_hold['trial_id'].max()],[0,0],'k--')
    sns.scatterplot(
        ax=lda_ax,
        data=td_hold,
        y='beh_lda',
        x='trial_id',
        hue=label_col,palette='Set1',hue_order=hue_order
    )
    lda_ax.text(0,-3,'Discriminability: {:.2f}'.format((td_hold['beh_pred']==td_hold[label_col]).mean()))
    lda_ax.set_ylabel('Behavioral LDA projection')
    lda_ax.set_xlabel('Trial ID')
    sns.despine(ax=lda_ax,trim=True)

    return lda_fig
