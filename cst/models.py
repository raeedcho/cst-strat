import numpy as np
import ssa

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

