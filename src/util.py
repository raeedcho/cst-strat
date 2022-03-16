import numpy as np
import pyaldata
import pandas as pd

def format_outfile_name(trial_data,postfix=''):
    '''
    Format a filename for output based on the trial_data structure
    
    Arguments:
        trial_data (DataFrame): PyalData formatted structure of neural/behavioral data
        postfix (str): postfix to add to the filename
        
    Returns:
        (str): formatted filename
    '''
    filename = '{monkey}_{session_date}_{postfix}'.format(
        monkey=trial_data['monkey'].values[0],
        session_date=np.datetime_as_string(trial_data['session_date'].values[0],'D').replace('-',''),
        postfix=postfix
    )

    return filename
    
def get_array_from_signal(signal):
    if signal.endswith('_spikes'):
        return signal.replace('_spikes','')
    elif signal.endswith('_rate'):
        return signal.replace('_rate','')
    else:
        raise ValueError('signal must end in "_spikes" or "_rate"')

@pyaldata.copy_td
def split_trials_by_epoch(trial_data,epoch_dict,epoch_col_name='epoch'):
    '''
    Split trial_data by epochs
    
    Arguments:
        trial_data (DataFrame): PyalData formatted structure of neural/behavioral data
        epoch_dict (dict): dictionary of epoch functions to pass into
            pyaldata.restrict_to_interval, with keys as epoch names
        epoch_col_name (str): name of column to add to trial_data with epoch names
        
    Returns:
        DataFrame: trial_data with entries for each specified epoch of each trial
    '''
    td_epoch_list = []
    for epoch_name,epoch_fun in epoch_dict.items():
        temp_td = pyaldata.restrict_to_interval(
            trial_data,
            epoch_fun=epoch_fun,
        )
        temp_td[epoch_col_name] = epoch_name
        td_epoch_list.append(temp_td)
    
    td_epoch = pd.concat(td_epoch_list).reset_index(drop=True)

    return td_epoch

def generate_realtime_epoch_fun(start_point_name,end_point_name=None,rel_start_time=0,rel_end_time=0):
    """
    Return a function that slices a trial around/between time points (noted in real time, not bins)

    Parameters
    ----------
    start_point_name : str
        name of the time point around which the interval starts
    end_point_name : str, optional
        name of the time point around which the interval ends
        if None, the interval is created around start_point_name
    rel_start_time : float, default 0
        when to start extracting relative to the starting time point
    rel_end_time : float, default 0
        when to stop extracting relative to the ending time point (bin is not included)

    Returns
    -------
    epoch_fun : function
        function that can be used to extract the interval from a trial
    """
    if end_point_name is None:
        epoch_fun = lambda trial: pyaldata.slice_around_point(
            trial,
            start_point_name,
            -rel_start_time/trial['bin_size'],
            rel_end_time/trial['bin_size']-1
        )
    else:
        epoch_fun = lambda trial: pyaldata.slice_between_points(
            trial,
            start_point_name,
            end_point_name,
            -rel_start_time/trial['bin_size'],
            rel_end_time/trial['bin_size']-1
        )

    return epoch_fun

def combine_all_trial_bins(td):
    '''
    Combine all bins in each trial into a single bin
    (wrapper on pyaldata.combine_time_bins)
    '''
    pass

def time_shuffle_columns(array):
    '''
    Shuffle independently within columns of numpy array

    Arguments:
        array (np.array): array to shuffle

    Returns:
        np.array: shuffled array
    '''
    rng = np.random.default_rng()

    for colnum in range(array.shape[1]):
        array[:,colnum] = rng.permutation(array[:,colnum])

    return array
    
def random_array_like(array):
    '''
    Returns an array of the same size as input,
    with the same overall mean and standard deviation
    (assuming a normal distribution)
    
    Arguments:
        array (np.array): array to imitate
        
    Returns:
        np.array: array with same mean and standard deviation
    '''
    rng = np.random.default_rng()
    return rng.standard_normal(array.shape) * np.std(array) + np.mean(array)

def form_neural_tensor(td,signal,cond_cols=None):
    '''
    Form a tensor of neural data from a trial_data structure
    Notes:
        - Trials must be the same length

    Arguments:
        td (DataFrame): trial_data structure in PyalData format
        signal (str): name of signal to form tensor from
        cond_cols (str or list): list of columns to use as conditions to group by
            If None, will form a third order tensor of shape (n_trials,n_neurons,n_timebins)

    Returns:
        np.array: tensor of neural data of shape (n_trials,n_neurons,n_cond_1,n_cond_2,n_cond_3,...,n_timebins)
    '''
    # Argument checking
    assert signal in td.columns, 'Signal must be in trial_data'
    assert pyaldata.trials_are_same_length(td), 'Trials must be the same length'

    if cond_cols is None:
        neural_tensor = np.stack([sig.T for sig in td[signal]],axis=0)
    else:
        td_grouped = td.groupby(cond_cols)
        min_trials = td_grouped.size().min()
        trial_cat_table = td_grouped.agg(
            signal = (signal, lambda sigs: np.stack([sig.T for sig in sigs.sample(n=min_trials)],axis=0))
        )
        # Stack in the remaining axes by iteratively grouping by remaining columns and stacking
        while trial_cat_table.index.nlevels > 1:
            # group by all columns other than the first one
            part_group = trial_cat_table.groupby(level=list(range(1,trial_cat_table.index.nlevels)))
            # stack over the first column and overwrite the old table, keeping time axis at the end
            trial_cat_table = part_group.agg(
                signal = ('signal', lambda sigs: np.stack(sigs,axis=-2))
            )
        else:
            # final stack
            neural_tensor = np.stack(trial_cat_table['signal'],axis=-2)

    return neural_tensor