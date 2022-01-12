import numpy as np
import pyaldata
import pandas as pd

def get_task_epoch_neural_averages(trial_data,hold_start=-0.4):
    '''
    Takes trial data as input and returns a table of average firing rates per neuron,
    separated by array, task (CO or CST), and epoch (hold or move)

    Arguments:
        trial_data (DataFrame): PyalData formatted structure of neural/behavioral data
        hold_start (float): time (in sec) relative go cue to consider the hold time for CO and CST trials
            (default: -0.4)

    Returns:
        DataFrame: table of average firing rates per neuron, with columns:
            ['monkey','session_date','task','epoch','array','chan_id','unit_id','average_rate']
    '''
    array_names = [col.replace('_spikes','') for col in trial_data if col.endswith('_spikes')]
    tasks = ['CO', 'CST']
    epochs = ['hold', 'move']

    fr_table_list = []
    for array in array_names:
        # ensure that we're dealing with spike counts here
        assert pyaldata.all_integer(np.concatenate(trial_data[array+'_spikes'].values))

        unit_guide = trial_data.loc[trial_data.index[0],array+'_unit_guide']

        for task in tasks:
            for epoch in epochs:
                temp_td = trial_data.loc[trial_data['task']==task,:].copy()
                if epoch == 'hold':
                    temp_td = pyaldata.restrict_to_interval(
                        temp_td,
                        start_point_name='idx_goCueTime',
                        rel_start=hold_start//temp_td['bin_size'].values[0],
                        rel_end=0,
                        reset_index=False
                    )
                elif epoch == 'move' and task == 'CO':
                    temp_td = pyaldata.restrict_to_interval(
                        temp_td,
                        start_point_name='idx_goCueTime',
                        end_point_name='idx_otHoldTime',
                        reset_index=False
                    )
                elif epoch == 'move' and task == 'CST':
                    temp_td = pyaldata.restrict_to_interval(
                        temp_td,
                        start_point_name='idx_cstStartTime',
                        end_point_name='idx_cstEndTime',
                        reset_index=False
                    )

                avg_fr = pyaldata.get_average_firing_rates(temp_td,array+'_spikes',divide_by_bin_size=True)

        fr_table_list.append(pd.DataFrame({
            'monkey': temp_td['monkey'].values[0],
            'session_date': temp_td['session_date'].values[0],
            'task': task,
            'epoch': epoch,
            'array': array,
            'chan_id': unit_guide[:,0],
            'unit_id': unit_guide[:,1],
            'average_rate': avg_fr
        }))

    return pd.concat(fr_table_list)
