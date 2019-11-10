# This is a practice of using ED to solve 1D S=1/2 Heisenberg Model
# Follow the instruction of arXiv: 1102.4006v1
import numpy as np
import scipy as sp
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt

'''Tag functions'''
'''Note: We can use the corresponding decimal number of the basis states as
         its tag.'''
def is_prime(num):
    factor = 2
    while(factor * factor <= num):
        if num % factor == 0:
            return False
        factor += 1
    return True

def primes(M):
    '''return a list of the first M prime numbers'''
    count = 0
    num = 2
    primes = np.zeros(M)

    while (count < M):
        if is_prime(num):
            primes[count] = num
            count += 1
        num += 1
    return primes

def find_tag(vec, M):
    '''calculate the tag of each vectors'''
    prime = primes(M)
    tag = 0
    for i in range(M):
        tag += np.sqrt(prime[i]) * vec[i]
    return tag

'''Heisenberg model: H = J ∑_<ij> si ⋅ sj
   Number of sites: M
   Hilbert space dimension: 2^N
   basis : eigen-states of Sz
           spin up --> 1 , spin down --> 0
    '''

'''Rewrite Hamiltonian to two parts:
   Diagonal: Hz = Jz S^z S^z
   Off-diagonal: Hxy = Jxy/2 (S^+ S^-  +  S^- S^+)
                 S+ = S^x  + i S^y
                 S- = S^x  - i S^y
   '''


def basis_vecs(N):
    '''Build basis vectors'''
    dim = pow(2,N)
    basis = np.zeros([dim,N])
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

basis = basis_vecs(5)
print(basis)

def heisenberg(N):
    '''Build Heisenberg model Hamiltonian'''
    return


'''Add constrain: total Sz = S
'''
def basis_conserve_sz(N):
    '''Build basis of total sz = S'''
    return

def heisenberg_conserve_sz(N):
    '''Build Hamiltonian with conserved Sz = S'''
    return
