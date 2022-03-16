import numpy as np

# for SSA
import ssa

# for GPFA
import elephant.gpfa
import quantities as pq
import neo

class SSA(object):
    def __init__(self,R=None,lam=.01,lr=0.001,n_epochs=3000,verbose=True, scheduler_params_input=dict()):
        self.ssa_params = {
                'R': R,
                'lam': lam,
                'lr': lr,
                'n_epochs': n_epochs,
                'verbose': verbose,
                'scheduler_params_input': scheduler_params_input
                }

    def fit(self,X,sample_weight=None):
        self.model,self.latent,_= ssa.fit_ssa(X,sample_weight=sample_weight,**self.ssa_params)

    def transform(self,X):
        [X_torch] = ssa.torchify([X])
        latent,_ = self.model(X_torch)

        return latent.detach().numpy()

class GPFA(elephant.gpfa.GPFA):
    def __init__(self,**kwargs):
        # precondition initialization to add SI units to inputs if they're not there
        if 'bin_size' in kwargs and type(kwargs['bin_size']) is not pq.Quantity:
            kwargs['bin_size'] *= pq.s
        if 'tau_init' in kwargs and type(kwargs['tau_init']) is not pq.Quantity:
            kwargs['tau_init'] *= pq.s

        super().__init__(**kwargs)

    def fit(self,data):
        """
        Fits the gpfa model to data (converting inputs to allow use of elephant implementation of GPFA)
        Inputs:
            data - list of numpy arrays, where data[k] is the population neural spike train on trial k, binned at 1 ms.
                Each array is T_k x N, where T_k is the length of trial k and N is the number of neurons.
        """
        super().fit(_convert_binned_spikes_to_spiketrain(data))

    def transform(self,data,returned_data=['latent_variable_orth']):
        """
        Projects data to the gpfa model (converting inputs to allow use of elephant implementation of GPFA)
        Inputs:
            data - list of numpy arrays, where data[k] is the population neural spike train on trial k, binned at 1 ms.
                Each array is T_k x N, where T_k is the length of trial k and N is the number of neurons.
            returned_data - which types of data to return (check elephant.gpfa.GPFA.transform docs)
        """
        result = super().transform(_convert_binned_spikes_to_spiketrain(data),**kwargs)

        if returned_data.shape[0]>1:
            for var in returned_data:
                result[var] = result[var].T
        else:
            result = result.T

        return result

def _convert_binned_spikes_to_spiketrain(data):
    """
    Converts a list on numpy arrays with binned spikes into a list of neo.SpikeTrain (mainly for GPFA)
    Inputs:
        data - list of numpy arrays, where data[k] is the population neural spike train on trial k, binned at 1 ms.
            Each array is T_k x N, where T_k is the length of trial k and N is the number of neurons.
    """
    # convert data to neo.SpikeTrains
    trial_spikes = []
    for trial_arr in data:
        timevec = np.arange(trial_arr.shape[0])
        spiketrains = []
        for neuron_arr in trial_arr.T:
            spike_times = [
                time
                for time,bin_spike in zip(timevec,neuron_arr)
                if bin_spike>0
            ]
            spiketrains.append(neo.SpikeTrain(spike_times,units='sec',t_stop=timevec[-1]))

        trial_spikes.append(spiketrains)

    return trial_spikes
