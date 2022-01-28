import pandas as pd
import pyaldata
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import scipy
import k3d
import ipywidgets

def plot_sensorimotor(cursor_pos=[],hand_pos=[], ax=None, clear_ax=True, scatter_args=dict(),**kwargs):
    """ Make sensorimotor plot of hand position vs cursor position
    
    Inputs:
    (keyword)
        ax (None) - axis object to plot in (default behavior makes a new axis object)
        scatter_args (dict()) - dict of kwargs to pass into ax.scatter
    
    Outputs:
        sc - scatter plot data so animations can reset them
    """
    if ax is None:
        ax = plt.gca()

    if clear_ax is True:
        ax.clear()

    ax.plot([-60,60],[60,-60],'--k')
    ax.plot([0,0],[-60,60],'-k')
    ax.plot([-60,60],[0,0],'-k')
    sc = ax.scatter(cursor_pos,hand_pos,**scatter_args)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('Cursor position')
    ax.set_ylabel('Hand position')
    sns.despine(ax=ax,left=True,bottom=True)
    
    return sc

def plot_sensorimotor_velocity(cursor_vel=[],hand_vel=[], ax=None, clear_ax=True, scatter_args=dict(),**kwargs):
    """ Make sensorimotor plot of hand velocity vs cursor velocity
    
    Inputs:
    (positional)
        trial - pyaldata row to plot
    (keyword)
        ax (None) - axis object to plot in (default behavior makes a new axis object)
        scatter_args (dict()) - dict of kwargs to pass into ax.scatter
    
    Outputs:
        sc - scatter plot data so animations can reset them
    """
    if ax is None:
        ax = plt.gca()

    if clear_ax is True:
        ax.clear()

    ax.plot([-60,60],[60,-60],'--g')
    ax.plot([0,0],[-100,100],'--k')
    ax.plot([-60,60],[0,0],'-k')
    ax.fill_between([0,60],0,y2=100,color=[1,0.8,0.8])
    ax.fill_between([-60,0],0,y2=-100,color=[1,0.8,0.8])
    sc = ax.scatter(cursor_vel,hand_vel,**scatter_args)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('Cursor command')
    ax.set_ylabel('Hand velocity')
    sns.despine(ax=ax,left=True,bottom=True)
    
    return sc

def plot_sm_tangent_angle(trialtime=[],cursor_vel=[],hand_vel=[],ax=None,clear_ax=True,scatter_args=dict(),**kwargs):
    if ax is None:
        ax = plt.gca()
        
    if clear_ax is True:
        ax.clear()

    # plot out guidance lines for tangent angle interpretation
    ax.plot([0,6],[-90,-90],'--k')
    ax.plot([0,6],[90,90],'--k')
    ax.plot([0,6],[-45,-45],'--g')
    ax.plot([0,6],[135,135],'--g')
    ax.fill_between([0,6],0,y2=90,color=[1,0.8,0.8])
    ax.fill_between([0,6],-180,y2=-90,color=[1,0.8,0.8])
    
    # actual tangent angle plot
    sc = ax.scatter(
        trialtime,
        np.arctan2(hand_vel,cursor_vel)*180/np.pi,
        **scatter_args
    )
    
    ax.set_ylabel('SM tangent angle')
    ax.set_xticks([0,6])
    ax.set_yticks([-180,-90,0,90,180])
    
    sns.despine(ax=ax,left=False,trim=True)
    
    return sc

def plot_sm_tangent_magnitude(trialtime=[],cursor_vel=np.array([]),hand_vel=np.array([]),ax=None,clear_ax=True,scatter_args=dict(),**kwargs):
    if ax is None:
        ax = plt.gca()
        
    if clear_ax is True:
        ax.clear()

    # sensorimotor tangent magnitude
    sc = ax.scatter(
        trialtime,
        hand_vel**2 + cursor_vel**2,
        **scatter_args
    )
    ax.set_xlabel('Time (s)')
    ax.set_ylabel("SM 'energy'")
    ax.set_xticks(np.arange(7))
    
    sns.despine(ax=ax,left=False,trim=True)
    
    return sc

def plot_cst_monitor_instant(cursor_pos=0,hand_pos=0,ax=None,clear_ax=True,**kwargs):
    if ax is None:
        ax = plt.gca()
        
    if clear_ax is True:
        ax.clear()

    ax.fill_between(
        [-50,50],
        0,y2=2,
        color=[0.5,0.5,0.5]
    )
    ax.set_xlim(-60,60)
    ax.set_ylim(-2,2)
    ax.set_xticks([])
    ax.set_yticks([])
    cursor_sc = ax.scatter(0,1,s=100,c='b')
    hand_sc = ax.scatter(0,-1,s=100,c='r')
    sns.despine(ax=ax,left=True,bottom=True)
    
    return cursor_sc,hand_sc

def plot_cst_traces(trialtime=None,cursor_pos=np.array([]),hand_pos=np.array([]),ax=None,clear_ax=True,flipxy=False,**kwargs):
    if ax is None:
        ax = plt.gca()
        
    if clear_ax is True:
        ax.clear()

    if trialtime is None:
        trialtime=np.arange(hand_pos.shape[0])
        
    if flipxy:
        ax.plot([0,0],[0,6],'-k')
        ax.set_ylim(0,6)
        ax.set_xlim(-60,60)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel('Time (s)')
        ax.set_xlabel('Cursor or hand position')
        sns.despine(ax=ax,left=True,bottom=False,trim=True)
        cursor_l, = ax.plot(cursor_pos,trialtime,'-b')
        hand_l, = ax.plot(hand_pos,trialtime,'-r')
    else:
        ax.plot([0,6],[0,0],'-k')
        ax.set_ylim(-60,60)
        ax.set_xlim(0,6)
        ax.set_yticks([-50,50])
        ax.set_xticks([])
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Cursor or hand position')
        sns.despine(ax=ax,left=False,bottom=True,trim=True)
        cursor_l, = ax.plot(trialtime,cursor_pos,'-b')
        hand_l, = ax.plot(trialtime,hand_pos,'-r')
        
    return cursor_l,hand_l

def make_behavior_neural_widget(trial_data,behavior_signals=None,neural_signals=None):
    """
    Make widgets to explore behavior and neural activity of trial data simultaneously
    Inputs:
        trial_data - pandas DataFrame with trial_data structure
        behavior_signals - callable function to extract behavioral signals from single trial
            (sigs = behavior_signals(trial))
        neural_signals - callable function to extract neural signals from single trial
            (sigs = neural_signals(trial))
    """

    assert callable(behavior_signals) and callable(neural_signals), 'behavior and neural signals inputs must be callable functions!'
    behavior_plot = k3d.plot(name='Behavioral plot')
    behavior_line = k3d.line(
        np.array([0,0,0],dtype=np.float32),
        attribute=np.float32(6),
        color_map=k3d.matplotlib_color_maps.Viridis,
        color_range=[0,6],
        shader='mesh',
        width=0.25,
    )
    behavior_plot+=behavior_line
    
    trace_plot = k3d.plot(name='Neural traces',camera_fov=40)
    trace_line = k3d.line(
        np.array([0,0,0],dtype=np.float32),
        attribute=np.float32(6),
        color_map=k3d.matplotlib_color_maps.Viridis,
        color_range=[0,6],
        shader='mesh',
        width=0.25,
    )
    trace_plot+=trace_line
    
    @ipywidgets.interact(trial_id=list(trial_data.index))
    def plot_trace(trial_id):
        trial = trial_data.loc[trial_id,:]
        # plot traces
        behavior_line.vertices = behavior_signals(trial)
        behavior_line.attribute = trial['trialtime'].astype(np.float32)

        trace_line.vertices = neural_signals(trial)
        trace_line.attribute = trial['trialtime'].astype(np.float32)
        
    behavior_plot.display()
    trace_plot.display()

def plot_horizontal_hand_movement(trial, ax=None, events=None, ref_event_idx=0):
    '''
    Plot out horizonatal hand movement against time, with some events marked

    Arguments:
        trial - pandas Series with trial_data structure (a row of trial_data)
        ax - matplotlib axis to plot into (default: None)
        events - list of event times to plot (default: None)
        ref_event_idx - event index to reference the trial time to (default: 0)

    Returns:
        ax - matplotlib axis with traces plotted
    '''
    if ax is None:
        ax = plt.gca()
        
    ax.clear()
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Hand position (mm)')

    trialtime = np.arange(trial['rel_hand_pos'].shape[0])

    ax.plot(trialtime[[0,-1]], [0,0], '-k')
    ax.plot(trialtime, trial['rel_hand_pos'][:,0],'-r')

    if events is not None:
        for event in events:
            event_time = trial[event]
            ax.plot(event_time*np.array([1,1]),[-50,50],'--g')

    scale_idx_ticks_to_time(ax.xaxis,trial['bin_size'],ref_event_idx)
    ax.set_title('Horizontal hand movement')
    sns.despine(ax=ax,trim=True,offset=10)

    return ax

def make_trial_raster(trial, ax=None, events=None, ref_event_idx=0):
    '''
    Make a raster plot for a given trial

    Arguments:
        trial - pandas Series with trial_data structure (a row of trial_data)
        ax - matplotlib axis to plot into (default: None)
        events - list of event times to plot (default: None)
        ref_event_idx - event index to reference the trial time to (default: 0)

    Returns:
        ax - matplotlib axis with traces plotted
    '''
    if ax is None:
        ax = plt.gca()
        
    ax.clear()
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Neurons')
    
    if trial['bin_size'] > 0.01:
        # use an image to plot the spikes
        ax.imshow(
            trial['M1_spikes'].T,
            aspect='auto',
            cmap='binary',
            origin='lower',
        )
    else:
        # use sparse matrix plot
        ax.spy(
            trial['M1_spikes'].T,
            aspect='auto',
            marker='|',
            markerfacecolor='k',
            markeredgecolor='k',
            markersize=2,
            origin='lower',
        )
        ax.tick_params(axis='x', top=False)

    if events is not None:
        for event in events:
            event_time = trial[event]
            ax.plot(event_time*np.array([1,1]),[0,trial['M1_spikes'].shape[1]],'--g')

    scale_idx_ticks_to_time(ax.xaxis,trial['bin_size'],ref_event_idx)
    ax.set_yticks([])
    ax.set_title('Neural raster')
    sns.despine(ax=ax,left=True,bottom=False,trim=True,offset=10)

    return ax

def scale_idx_ticks_to_time(axis,bin_size,ref_event_idx=0):
    '''
    TEMPORARY FUNCTION TO SCALE TICKS TO TIME
    Scale the index ticks of a matplotlib axis to time
    (Note: this doesn't change any data in the plot--it just changes the tick labels)
    Side effects: (intended) changes prperties of the axis passed in
    TODO: figure out how to change extent of spy plots to make this function unneeded

    Arguments:
        axis - matplotlib axis to change tick labels of (e.g. plt.gca().xaxis)
        bin_size - correspondence between index and time (size of bin in seconds)
        ref_event_idx - index of event to reference the trial time to (default: 0)
    '''
    ticks = mpl.ticker.FuncFormatter(lambda x,pos: '{0:g}'.format((x-ref_event_idx)*bin_size))
    axis.set_major_formatter(ticks)

# def plot_slds_fit(trial):
#     '''
#     Plots out the SLDS fit for a given trial. They are in order, from top to bottom:
#     - Behavioral kinematics plot
#     - discrete state plot
#     - continuous state plot
#     - emmisions (neural firing) raster
#     
#     Note: this function is unfinished
#     '''
#     fig.clear()   
#     [beh_ax,disc_ax,cont_ax,emis_ax] = fig.subplots(4,1)
#     
#     hand_pos = trial['rel_hand_pos']
#     lim = abs(hand_pos[:2]).max()
#     for d in range(2):
#         beh_ax.plot(hand_pos[:,d]+lim*d,'-k')
#     beh_ax.plot(trial['idx_goCueTime']*np.array([1,1]),lim*np.array([-1,1]),'--r')
#     beh_ax.set_xticks([])
#     beh_ax.set_yticks(np.arange(2) * lim)
#     beh_ax.set_yticklabels(['horizontal','vertical'])
#     beh_ax.set_title('Hand position')
#     sns.despine(ax=beh_ax,trim=True,bottom=True)
#     
#     cmap_limited = mpl.colors.ListedColormap(colors[0:num_disc_states])
#     disc_ax.imshow(trial['M1_slds_disc'][None,:],aspect='auto',cmap=cmap_limited)
#     disc_ax.plot(trial['idx_goCueTime']*np.array([1,1]),[-0.5,0.5],'--r')
#     disc_ax.set_yticks([])
#     disc_ax.set_xticks([])
#     disc_ax.set_title('Most likely discrete state')
#     
#     sns.despine(ax=disc_ax,left=True,bottom=True)
# 
#     lim = abs(trial['M1_slds_cont']).max()
#     for d in range(latent_dim):
#         cont_ax.plot(trial['M1_slds_cont'][:,d],'-k')
#     cont_ax.plot(trial['idx_goCueTime']*np.array([1,1]),lim*np.array([-1,1]),'--r')
#     # cont_ax.set_yticks(np.arange(latent_dim) * lim)
#     # cont_ax.set_yticklabels(["$x_{}$".format(d+1) for d in range(latent_dim)])
#     cont_ax.set_xticks([])
#     cont_ax.set_title('Estimated continuous latents')
#     sns.despine(ax=cont_ax,bottom=True,trim=True)
# 
#     emis_ax.imshow(trial['M1_spikes'].T,aspect='auto',cmap='inferno')
#     emis_ax.plot(trial['idx_goCueTime']*np.array([1,1]),[0,trial['M1_spikes'].shape[1]],'--r')
#     emis_ax.set_yticks([])
#     emis_ax.set_title('Emissions raster')
#     sns.despine(ax=emis_ax,trim=True)