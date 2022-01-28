import numpy as np

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