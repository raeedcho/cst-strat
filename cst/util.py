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