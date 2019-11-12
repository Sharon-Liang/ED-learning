# This is a practice of using ED to solve 1D S=1/2 Heisenberg Model
# Follow the instruction of arXiv: 1102.4006v1
import numpy as np
import scipy as sp
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from scipy.special import comb, perm #Combination and Permulation operation

'''Tag functions'''
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

'''Note: We can use the corresponding decimal number of the basis states as
         its tag.'''
def decimal(vec):
    '''Find the corresponding decimal number of vec'''
    N = len(vec)
    dec = 0
    for i in range(N):
        dec += vec[i] * pow(2,N-1-i)
    return dec

'''Heisenberg model: H = J ∑_<ij> si ⋅ sj
   Number of sites: M
   Hilbert space dimension: 2^N
   basis : eigen-states of Sz
           spin up --> 1 , spin down --> 0
    '''

def basis_vecs(N):
    '''Build basis vectors'''
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


'''Rewrite Hamiltonian to two parts:
   Diagonal: Hz = Jz S^z S^z
   Off-diagonal: Hxy = Jxy/2 (S^+ S^-  +  S^- S^+)
                 S+ = S^x  + i S^y
                 S- = S^x  - i S^y
   '''

def heisenberg(N, Jz, Jxy, h, periodic = False, hermition = False):
    '''Build Heisenberg model Hamiltonian'''
    dim = pow(2,N)
    basis = basis_vecs(N)
    if periodic:
        limit = N
    else:
        limit = N-1
    # Diagonal: Hz = Jz S^z S^z
    Hz = np.zeros([dim,dim])
    for i in range(dim):
        for j in range(limit):
            if j == N-1:
                next = 0
            else:
                next = j+1
            Hz[i,i] += - Jz * (basis[i,j] - 1/2) * (basis[i,next] - 1/2)

    #Off-diagonal: Hxy = Jxy/2 (S^+ S^-  +  S^- S^+)
    Hxy = np.zeros([dim,dim])
    for i in range(dim):
        for j in range(limit):
            if j == N-1:
                next = 0
            else:
                next = j + 1
            vec = np.zeros(N,int)
            vec[j] = 1
            vec[next] = 1
            if basis[i,j]!= basis[i,next]:
                new = basis[i,:].copy()
                for k in range(N):
                    new[k] = vec[k]^new[k]
                Hxy[i,decimal(new)] += - Jxy / 2
    # add small h to break degeneracy
    Hh = np.zeros([dim,dim])
    for i in range(dim):
        for j in range(N):
            Hh[i,i] += h * (basis[i,j] - 1/2)
    H = Hz + Hxy + Hh
    if hermition:
        H = (H + np.conj(H).T)/2
    return H

'''Add constrain: total Sz = S
   For a length = N(N=even) chain, there are N/2 spin-up.
'''

'''The idea is:(conserved particle number problem)
   1. Separate the chain to two parts, one part has M spin-up,
      the other has N/2-M spin-up.
   2. Determin two part separately. We only have to figure out the spin
      congigurations of M particles in N/2 sites(M <= N/2)
   '''
def half_chain(N,M):
    '''M spin-up on N sites: M<=N'''
    dim = int(comb(N,M))
    vecs = np.zeros([dim,N],int)
    if M == 0:
        return vecs
    elif 0 < M <= N:
        for i in range(M):
            vecs[0,i] = 1
        for j in range(dim-1):
            k = N-2
            for i in range(N-2,-1,-1):
                if vecs[j,i]==0:
                    k -= 1
                    if k >= 0:
                        continue
                    else:
                        print('Error: zero vector!')
                elif vecs[j,i] != 0:
                    if vecs[j,i+1]!=0:
                        k -=1
                    else:
                        break

            for i in range(k):
                vecs[j+1,i] = vecs[j,i]

            vecs[j+1,k] = vecs[j,k] - 1
            vecs[j+1,k+1] = vecs[j,k+1] + 1

            sum = 0
            for i in range(k+2):
                if vecs[j,i] == 1:
                    sum += 1
            sum = M - sum
            if sum > 0:
                for i in range(k+2,k+2+sum):
                    vecs[j+1,i] = 1
    return vecs

vec = half_chain(6,4)
print(vec)


def basis_conserve_sz(N,M):
    '''Build basis of total Sz = M/2
       M: numer of spin-up, if M <= 1/2'''
    dim = basis_dim(N,M)


    return

def heisenberg_conserve_sz(N):
    '''Build Hamiltonian with conserved Sz = S'''
    return
