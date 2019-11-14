# This is a practice of using ED to solve 1D S=1/2 Heisenberg Model
# Follow the instruction of arXiv: 1102.4006v1
import numpy as np
import scipy as sp
import scipy.stats as spst #Statistical functions
import matplotlib.pyplot as plt

from scipy.sparse.linalg import eigsh
from scipy.special import comb, perm #Combination and Permulation operation

#-------------------------------------------------------------------------------
'''Helper functions: help with visualization and debugging'''
def basisVisualizer(L,psi):
    '''Given the dicimal tag of the state psi, outputs the state in arrows
       L: total length of the chain'''
    #ex: |↓|↑|↓|↑|↑|
    psi_2 = bin(psi)[2:] # Alternative: psi_2 = format(psi,'b')
    N  = len(psi_2)
    up = (L-N)*'0'+psi_2 # add two string
    configStr = "|"
    uparrow   = '\u2191'
    downarrow = '\u2193'
    for i in range(L):
        blank = True
        if up[i] == '1':
            configStr+=uparrow
            blank = False
        if up[i] == '0':
            configStr+=downarrow
            blank = False
        if blank:
            configStr+="_"
        configStr +="|"
    print(configStr)

def countBits(x):
    '''Counts number of 1s in bin(n)'''
    #From Hacker's Delight, p. 66
    x = x - ((x >> 1) & 0x55555555)
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F
    x = x + (x >> 8)
    x = x + (x >> 16)
    return x & 0x0000003F

#-------------------------------------------------------------------------------
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

def find_tag(vec):
    '''calculate the tag of each vectors'''
    M = len(vec)
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

#-------------------------------------------------------------------------------
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

def heisenberg(N, Jz, Jxy, h, periodic = False):
    '''Build Heisenberg model Hamiltonian'''
    basis = basis_vecs(N)
    dim = len(basis)
    if periodic:
        limit = N
    else:
        limit = N-1

    Hz = np.zeros([dim,dim]) # Diagonal: Hz = Jz S^z S^z
    Hxy = np.zeros([dim,dim]) #Off-diagonal: Hxy = Jxy/2 (S^+ S^-  +  S^- S^+)
    Hh = np.zeros([dim,dim]) # add small h to break degeneracy

    for i in range(dim):
        for j in range(limit):
            next = (j+1) % N
            Hz[i,i] += - Jz * (basis[i,j] - 1/2) * (basis[i,next] - 1/2)

            vec = np.zeros(N,int)
            vec[j] = 1
            vec[next] = 1
            if basis[i,j]!= basis[i,next]:
                new = basis[i,:].copy()
                for k in range(N):
                    new[k] = vec[k]^new[k]

                Hxy[i,decimal(new)] += - Jxy / 2

            Hh[i,i] += h * (basis[i,j] - 1/2)

    H = Hz + Hxy + Hh
    return H

#-------------------------------------------------------------------------------
'''Add constrain: total Sz = S
   For a length = N(N=even) chain, there are N/2 spin-up.
'''

'''The idea is:(conserved particle number problem)
   1. Separate the chain to two parts, one part has M spin-up,
      the other has N/2-M spin-up.
   2. Determin two part separately. We only have to figure out the spin
      congigurations of M particles in N/2 sites(M <= N/2)
   '''
def basis_sz_conserve(N,M):
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

def heisenberg_sz_conserve(N, Jz, Jxy, Sz, periodic = False):
    '''Build Hamiltonian with conserved Sz = S'''
    M = N/2 + Sz
    if M%1 != 0:
        return print('Error: No such configuration')
    else:
        M = int(M)
    basis = basis_sz_conserve(N,M)
    dim = len(basis)

    # Define a tag for every state
    T = np.zeros(dim)
    for i in range(dim):
        T[i] = find_tag(basis[i,:])
    Tsorted = sorted(T)
    ind = np.argsort(T)  # Tsorted[i] = T[ind[i]]

    if periodic:
        limit = N
    else:
        limit = N-1

    Hz = np.zeros([dim,dim]) # Diagonal: Hz = Jz S^z S^z
    Hxy = np.zeros([dim,dim]) #Off-diagonal: Hxy = Jxy/2 (S^+ S^-  +  S^- S^+)

    for i in range(dim):
        for j in range(limit):
            next = (j+1) % N
            Hz[i,i] += - Jz * (basis[i,j] - 1/2) * (basis[i,next] - 1/2)

            vec = np.zeros(N,int)
            vec[j] = 1
            vec[next] = 1
            if basis[i,j]!= basis[i,next]:
                new = basis[i,:].copy()
                for k in range(N):
                    new[k] = vec[k]^new[k]
                tag = find_tag(new) # tag for the newly generated vector
                # Find the position of the new vector
                for k in range(N):
                    if tag == Tsorted[k]:
                            Hxy[i,ind[k]] += - Jxy / 2
    H = Hz + Hxy
    return H

#-------------------------------------------------------------------------------
'''Eigenvalue and Eigenvectors'''
H = heisenberg(4, -1, -1, 0, periodic = True)
ev, evec = np.linalg.eigh(H)
print(ev)
