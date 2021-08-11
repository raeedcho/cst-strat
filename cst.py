import pandas as pd
import pyaldata
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import scipy

def load_clean_data(filename):
    td = pyaldata.mat2dataframe(filename,True,'trial_data')

    # condition dates and times
    td['date_time'] = pd.to_datetime(td['date_time'])
    td['session_date'] = pd.DatetimeIndex(td['date_time']).normalize()

    # add trial time to td
    td['trialtime'] = [trial['bin_size']*np.arange(trial['hand_pos'].shape[0]) for (_,trial) in td.iterrows()]

    # # separate out non-time-varying fields into a separate trial table
    # timevar_cols = td.columns.intersection(pyaldata.get_time_varying_fields(td))
    # trial_info = td.drop(columns=timevar_cols)
    # timevar_data = td[timevar_cols]

    # # melt out time information in time-varying column dataframe
    # for (idx,trial) in timevar_data.iterrows():
    #     #split out rows of numpy array
    #     temp = pd.DataFrame({key: list(val_array) for (key,val_array) in trial.iteritems()})

    #     # add a timedelta column to DataFrame
    #     temp['trialtime'] = pd.to_timedelta(trial_info.loc[idx,'bin_size']*np.arange(trial[timevar_cols[0]].shape[0]))

    #     # add trial index columns (have to figure out what to do about unit guide...)
    #     for col in trial_info.columns:
    #         temp[col] = trial_info.loc[idx,col]
    # 
    # # set up a multi-index for trials
    # td.set_index(['monkey','session_date','trial_id'],inplace=True)

    # remove aborts
    abort_idx = np.isnan(td['idx_goCueTime']);
    td = td[~abort_idx]
    print(f"Removed {sum(abort_idx)} trials that monkey aborted")

    # trim nans off the end of trials for hand pos
    for trial_idx, trial in td.iterrows():
        nan_times = np.any(np.isnan(trial['hand_pos']),axis=1)
        first_viable_time = np.nonzero(~nan_times)[0][0]
        last_viable_time = np.nonzero(~nan_times)[0][-1]
        epoch_fun = lambda x: slice(first_viable_time,last_viable_time+1)
        new_trial = pyaldata.restrict_to_interval(
                trial.to_frame().T,
                epoch_fun=epoch_fun
                ).squeeze()
        td.loc[trial_idx,:] = new_trial

    # fill kinematic signals
    cutoff = 30 # Hz
    samprate = 1/td.at[td.index[0],'bin_size'];
    filt_b, filt_a = scipy.signal.butter(4,cutoff/(samprate/2));
    td['hand_pos'] = [scipy.signal.filtfilt(filt_b,filt_a,signal,axis=0) for signal in td['hand_pos']]
    td = pyaldata.add_gradient(td,'hand_pos',outfield='hand_vel',normalize=True)
    td = pyaldata.add_gradient(td,'hand_vel',outfield='hand_acc',normalize=True)
    td = pyaldata.add_gradient(td,'cursor_pos',outfield='cursor_vel',normalize=True)

    # neural data considerations
    unit_guide = td.loc[td.index[0],'M1_unit_guide']
    if unit_guide.shape[0]>0:
        bad_units = unit_guide[:,1]<=1;

        # for this particular file only, remove correlated neurons...
        if td.loc[td.index[0],'session_date']==pd.to_datetime('2018/06/26'):
            corr_units = np.array([[8,2],[64,2]])
            bad_units = bad_units | (np.in1d(unit_guide[:,0],corr_units[:,0]) & np.in1d(unit_guide[:,1],corr_units[:,1]))
        
        td['M1_spikes'] = [spikes[:,~bad_units] for spikes in td['M1_spikes']]
        td['M1_unit_guide'] = [guide[~bad_units,:] for guide in td['M1_unit_guide']]

        td = pyaldata.remove_low_firing_neurons(td,'M1_spikes',0.1,divide_by_bin_size=True,verbose=True)

    return td

def plot_sensorimotor(cursor_pos=[],hand_pos=[], ax=None, scatter_args=dict(),**kwargs):
    """ Make sensorimotor plot of hand position vs cursor position
    
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

def plot_sensorimotor_velocity(cursor_vel=[],hand_vel=[], ax=None, scatter_args=dict(),**kwargs):
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

def plot_sm_tangent_angle(trialtime=[],cursor_vel=[],hand_vel=[],ax=None,scatter_args=dict(),**kwargs):
    if ax is None:
        ax = plt.gca()
        
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

def plot_sm_tangent_magnitude(trialtime=[],cursor_vel=np.array([]),hand_vel=np.array([]),ax=None,scatter_args=dict(),**kwargs):
    if ax is None:
        ax = plt.gca()
        
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

def plot_cst_monitor_instant(cursor_pos=0,hand_pos=0,ax=None,**kwargs):
    if ax is None:
        ax = plt.gca()
        
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

def plot_cst_traces(trialtime=[],cursor_pos=[],hand_pos=[],ax=None,flipxy=False,**kwargs):
    if ax is None:
        ax = plt.gca()
        
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
