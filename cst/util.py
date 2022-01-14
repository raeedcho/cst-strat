

def get_array_from_signal(signal):
    if signal.endswith('_spikes'):
        return signal.replace('_spikes','')
    elif signal.endswith('_rate'):
        return signal.replace('_rate','')
    else:
        raise ValueError('signal must end in "_spikes" or "_rate"')