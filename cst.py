import mat73
import pandas as pd

def get_cst_dataframe(filename):
    mat = mat73.loadmat(filename)
    trial_data = mat.trial_data
    trial_data = {
            'monkey': trial_data['monkey'],
            'date_time': trial_data['date_time'],
            'task': trial_data['task'],
            'trial_id': trial_data['trial_id'],
            'result': trial_data['result'],
            'bin_size': trial_data['bin_size'],
            'lambda': trial_data['lambda'],
            'ct_location': trial_data['ct_location'],
            'ot_location': trial_data['ot_location'],
            'idx_startTime': trial_data['idx_startTime'],
            'idx_endTime': trial_data['idx_endTime'],
            'idx_ctHoldTime': trial_data['idx_ctHoldTime'],
            'idx_goCueTime': trial_data['idx_goCueTime'],
            'idx_otHoldTime': trial_data['idx_otHoldTime'],
            'idx_rewardTime': trial_data['idx_rewardTime'],
            'idx_failTime': trial_data['idx_failTime'],
            'cursor_pos': trial_data['cursor_pos'],
            'hand_pos': trial_data['hand_pos'],
            'M1_unit_guide': trial_data['M1_unit_guide'],
            'M1_spikes': trial_data['M1_spikes'],
            }

    return pd.DataFrame.from_dict(trial_data)
