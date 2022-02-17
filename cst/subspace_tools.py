# a set of tools to manipulate and analyze geometry of high dimensional data

import numpy as np
from sklearn.decomposition import PCA

def subspace_overlap_index(X,Y,num_dims=10):
    '''
    Calculate the subspace overlap index (from Elsayed et al. 2016)
    between two data matrices (X and Y), given a number of dimensions

    Arguments:
        X,Y - (numpy arrays) matrix containing data (e.g. firing rates)
            with features along columns and observations along rows
        num_dims - (int) number of dimensions to use in the subspace
            overlap calculation

    Returns:
        (float) subspace overlap index
    '''
    assert X.shape[1] == Y.shape[1], 'X and Y must have same number of features'
    assert num_dims <= X.shape[1], 'num_dims must be less than or equal to number of features in X'
    assert X.ndim == 2, 'X must be a 2D array'
    assert Y.ndim == 2, 'Y must be a 2D array'

    # mean subtract X and Y
    X_hat = X - X.mean(axis=0)
    Y_hat = Y - Y.mean(axis=0)

    # get PCA of X and Y
    pca_X = PCA(n_components=num_dims)
    pca_Y = PCA(n_components=num_dims)
    pca_X.fit(X_hat)
    pca_Y.fit(Y_hat)

    # calculate overlap
    X_var = np.sum(pca_X.explained_variance_)
    X_cov = np.cov(X_hat.T)
    Y_axes = pca_Y.components_
    soi = np.trace(Y_axes @ X_cov @ Y_axes.T)/X_var

    return soi