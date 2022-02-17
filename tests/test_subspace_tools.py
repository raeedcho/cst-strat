
import pytest
import cst.subspace_tools as subspace_tools
import numpy as np

'''
Test subspace_overlap_index function with a few test cases:
1) X and Y are the same
2) X and Y occupy the same subspace
3) X and Y occupy different (random) subspaces
4) X and Y occupy orthogonal subspaces
5) X and Y are completely different and high dimensional
'''
def test_subspace_overlap_same_input():
    X_sub = np.random.rand(1000,10)
    proj_mat = np.random.rand(10,80)
    X = X_sub @ proj_mat
    Y = X.copy()
    soi = subspace_tools.subspace_overlap_index(X,Y,num_dims=10)
    assert soi == pytest.approx(1.0)

def test_subspace_overlap_same_subspace():
    X_sub = np.random.rand(1000,10)
    Y_sub = np.random.rand(1000,10)
    proj_mat = np.random.rand(10,80)
    X = X_sub @ proj_mat
    Y = Y_sub @ proj_mat
    soi = subspace_tools.subspace_overlap_index(X,Y,num_dims=10)
    assert soi == pytest.approx(1.0)

def test_subspace_overlap_different_subspace():
    X_sub = np.random.rand(1000,10)
    Y_sub = np.random.rand(1000,10)
    X_proj_mat = np.random.rand(10,80)
    Y_proj_mat = np.random.rand(10,80)
    X = X_sub @ X_proj_mat
    Y = Y_sub @ Y_proj_mat
    soi = subspace_tools.subspace_overlap_index(X,Y,num_dims=10)
    assert soi >= 0 and soi <= 1

def test_subspace_overlap_orthogonal_subspace():
    X_sub = np.random.rand(1000,10)
    Y_sub = np.random.rand(1000,10)
    X_proj_mat = np.random.rand(10,80)
    Y_proj_mat = np.random.rand(10,80)
    Y_proj_mat -= Y_proj_mat @ X_proj_mat.T @ np.linalg.inv(X_proj_mat @ X_proj_mat.T) @ X_proj_mat
    X = X_sub @ X_proj_mat
    Y = Y_sub @ Y_proj_mat
    soi = subspace_tools.subspace_overlap_index(X,Y,num_dims=10)
    assert soi == pytest.approx(0.0)