import pandas as pd
import pyaldata
import numpy as np
import scipy

def load_clean_data(filename):
    td = pyaldata.mat2dataframe(filename,True,'trial_data')

    # condition dates and times
    td['date_time'] = pd.to_datetime(td['date_time'])
    td['session_date'] = pd.DatetimeIndex(td['date_time']).normalize()

    # add trial time to td
    td['trialtime'] = [trial['bin_size']*np.arange(trial['hand_pos'].shape[0]) for (_,trial) in td.iterrows()]

    # remove aborts
    abort_idx = np.isnan(td['idx_goCueTime']);
    td = td[~abort_idx]
    print(f"Removed {sum(abort_idx)} trials that monkey aborted")

    td = trim_nans(td)
    td = fill_kinematic_signals(td)

    # neural data considerations
    unit_guide = td.loc[td.index[0],'M1_unit_guide']
    if unit_guide.shape[0]>0:
        # remove unsorted neurons (unit number <=1)
        bad_units = unit_guide[:,1]<=1;

        # for this particular file only, remove correlated neurons...
        if td.loc[td.index[0],'session_date']==pd.to_datetime('2018/06/26'):
            corr_units = np.array([[8,2],[64,2]])
            bad_units = bad_units | (np.in1d(unit_guide[:,0],corr_units[:,0]) & np.in1d(unit_guide[:,1],corr_units[:,1]))
        
        # mask out bad neural data
        td['M1_spikes'] = [spikes[:,~bad_units] for spikes in td['M1_spikes']]
        td['M1_unit_guide'] = [guide[~bad_units,:] for guide in td['M1_unit_guide']]

        td = pyaldata.remove_low_firing_neurons(td,'M1_spikes',0.1,divide_by_bin_size=True,verbose=True)

    return td

def trim_nans(trial_data):
    """
    Trim nans off of end of trials when hand position wasn't recorded
    """
    def epoch_fun(trial):
        nan_times = np.any(np.isnan(trial['hand_pos']),axis=1)
        first_viable_time = np.nonzero(~nan_times)[0][0]
        last_viable_time = np.nonzero(~nan_times)[0][-1]
        return slice(first_viable_time,last_viable_time+1)

    return pyaldata.restrict_to_interval(trial_data,epoch_fun=epoch_fun)

@pyaldata.copy_td
def fill_kinematic_signals(td,cutoff=30):
    """
    Fill out kinematic signals by filtering hand position and differentiating
    Inputs:
        - td: PyalData file
        - cutoff: cutoff frequency (Hz) for low-pass filter on hand position (default: 30)
    """
    samprate = 1/td.at[td.index[0],'bin_size'];
    filt_b, filt_a = scipy.signal.butter(4,cutoff/(samprate/2));
    td['hand_pos'] = [scipy.signal.filtfilt(filt_b,filt_a,signal,axis=0) for signal in td['hand_pos']]
    td['hand_vel'] = [np.gradient(trial['hand_pos'],trial['bin_size'],axis=0) for _,trial in td.iterrows()]
    td['hand_acc'] = [np.gradient(trial['hand_vel'],trial['bin_size'],axis=0) for _,trial in td.iterrows()]
    td['cursor_vel'] = [np.gradient(trial['cursor_pos'],trial['bin_size'],axis=0) for _,trial in td.iterrows()]

    return td

@pyaldata.copy_td
def mask_neural_data(td,mask):
    """
    Returns a PyalData structure with neural units kept according to mask
    (Basically just masks out both the actual neural data and the unit guide simultaneously)
    """
    pass

@pyaldata.copy_td
def relationize_td(trial_data):
    """
    Split out trial info and time-varying signals into their own tables

    Returns (trial_info,signals)
        trial_info - DataFrame with info about individual trials
        signals - 'trialtime' indexed DataFrame with signal values
    """
    # start out by making sure that trial_id is index
    td = trial_data.set_index('trial_id')

    # separate out non-time-varying fields into a separate trial table
    timevar_cols = td.columns.intersection(pyaldata.get_time_varying_fields(td))
    trial_info = td.drop(columns=timevar_cols)
    timevar_data = td[timevar_cols].copy()

    # melt out time information in time-varying column dataframe
    signals = []
    for (idx,trial) in timevar_data.iterrows():
        #split out rows of numpy array
        signal_dict = {key: list(val_array.copy()) for (key,val_array) in trial.iteritems()}
        signal_dict['trial_id'] = idx
        temp = pd.DataFrame(signal_dict)

        # add a timedelta column to DataFrame
        # temp['trialtime'] = pd.to_timedelta(trial_info.loc[idx,'bin_size']*np.arange(trial[timevar_cols[0]].shape[0]))
        temp['trialtime'] = pd.to_timedelta(trial_info.loc[idx,'bin_size']*np.arange(trial[timevar_cols[0]].shape[0]),unit='seconds')

        signals.append(temp)

    signals = pd.concat(signals)
    signals.set_index(['trial_id','trialtime'],inplace=True)
    
    # set up a multi-index for trials
    # td.set_index(['monkey','session_date','trial_id'],inplace=True)

    return trial_info,signals