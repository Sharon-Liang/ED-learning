# ED: Kitaev model (24 sites)
import numpy as np
import scipy as sp
import scipy.stats as spst #Statistical functions
import matplotlib.pyplot as plt

from scipy.sparse.linalg import eigsh
from scipy.special import comb, perm #Combination and Permulation operation

#-------------------------------------------------------------------------------
'''We use the corresponding decimal number of the basis states as its tag.'''
def decimal(vec):
    '''Find the corresponding decimal number of vec'''
    N = len(vec)
    dec = 0
    for i in range(N):
        dec += vec[i] * pow(2,N-1-i)
    return dec

#-------------------------------------------------------------------------------
def basis_vecs(N):
    '''Build basis vectors
        N: Number of sites'''
    dim = pow(2,N)
    basis = np.zeros([dim,N],int)
    for i in range(dim-1):
        #determin the position of the first 0, start from the lowest digit
        for k in range(N-1,-1,-1):
            if basis[i,k] == 0:
                break
        #write the next basis
        for j in range(k):
            basis[i+1,j] = basis[i,j]
        basis[i+1,k] = 1
        if k + 1 < N:
            for j in range(k+1,N):
                basis[i+1,j] = 0
    return basis

#-------------------------------------------------------------------------------
'''24 sites Kitaev model'''
def Kitaev(N):
    return
