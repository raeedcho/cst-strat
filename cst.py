import mat73
import pandas as pd
import pyaldata
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np


def plot_sensorimotor(trial, scale=60, ax=None, scatter_args=dict()):
    """ Make sensorimotor plot of hand position vs cursor position
    
    Inputs:
    (positional)
        trial - pyaldata row to plot
    (keyword)
        scale (60) - scale at which to plot (ylim and xlim set to -scale to scale)
        ax (None) - axis object to plot in (default behavior makes a new axis object)
        scatter_args (dict()) - dict of kwargs to pass into ax.scatter
    """
    if ax is None:
        ax = plt.gca()

    ax.plot([-60,60],[60,-60],'--k')
    ax.plot([0,0],[-60,60],'-k')
    ax.plot([-60,60],[0,0],'-k')
    ax.scatter(
        trial['cursor_pos'][:,0],
        trial['hand_pos'][:,0],
        **scatter_args
        )
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-scale,scale)
    ax.set_ylim(-scale,scale)
    ax.set_xlabel('Cursor position')
    ax.set_ylabel('Hand position')
    sns.despine(ax=ax,left=True,bottom=True)
    
def plot_sensorimotor_velocity(trial, ax=None, scatter_args=dict()):
    """ Make sensorimotor plot of hand velocity vs cursor velocity
    
    Inputs:
    (positional)
        trial - pyaldata row to plot
    (keyword)
        scale (60) - scale at which to plot (ylim and xlim set to -scale to scale)
        ax (None) - axis object to plot in (default behavior makes a new axis object)
        scatter_args (dict()) - dict of kwargs to pass into ax.scatter
    """
    if ax is None:
        ax = plt.gca()

    ax.plot([-60,60],[60,-60],'--g')
    ax.plot([0,0],[-100,100],'--k')
    ax.plot([-60,60],[0,0],'-k')
    ax.fill_between([0,60],0,y2=100,color=[1,0.8,0.8])
    ax.fill_between([-60,0],0,y2=-100,color=[1,0.8,0.8])
    ax.scatter(
        trial['cst_cursor_command'][:,0],
        trial['hand_vel'][:,0],
        **scatter_args
        )
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('Cursor command')
    ax.set_ylabel('Hand velocity')
    sns.despine(ax=ax,left=True,bottom=True)
    
def plot_sm_tangent_polar(trial, scatter_args=dict()):
    fig,ax = plt.subplots(2,1,figsize=(5,5),sharex=True,sharey=False)
    
    # plot out guidance lines for tangent angle interpretation
    ax[0].plot(
        [trial['trialtime'][0],trial['trialtime'][-1]],
        [-90,-90],
        '--k'
    )
    ax[0].plot(
        [trial['trialtime'][0],trial['trialtime'][-1]],
        [90,90],
        '--k'
    )
    ax[0].plot(
        [trial['trialtime'][0],trial['trialtime'][-1]],
        [-45,-45],
        '--g'
    )
    ax[0].plot(
        [trial['trialtime'][0],trial['trialtime'][-1]],
        [135,135],
        '--g'
    )
    ax[0].fill_between(
        trial['trialtime'],
        0, y2=90,
        color=[1,0.8,0.8]
    )
    ax[0].fill_between(
        trial['trialtime'],
        -180, y2=-90,
        color=[1,0.8,0.8]
    )
    
    # actual tangent angle plot
    ax[0].scatter(
        trial['trialtime'],
        np.arctan2(trial['hand_vel'][:,0],trial['cst_cursor_command'][:,0])*180/np.pi,
        **scatter_args
    )
    
    ax[0].set_ylabel('SM tangent angle')
    ax[0].set_xticks([])
    ax[0].set_yticks([-180,-90,0,90,180])
    
    # sensorimotor tangent magnitude
    ax[1].scatter(
        trial['trialtime'],
        trial['hand_vel'][:,0]**2 + trial['cst_cursor_command'][:,0]**2,
        **scatter_args
    )
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('SM tangent magnitude')
    ax[1].set_xticks(np.arange(7))
    
    sns.despine(ax=ax[0],left=False,bottom=True,trim=True)
    sns.despine(ax=ax[1],left=False,trim=True)