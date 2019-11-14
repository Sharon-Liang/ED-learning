# This is a practice of using ED to solve 1D Bose-Hubbard Model
# Follow the instruction of arXiv: 1102.4006v1
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from scipy.sparse.linalg import eigsh

#-------------------------------------------------------------------------------
'''Hashing technique : define a unique tag for each basis vector.

    Instead of calculating the matrix elements one by one, we apply the
    Hamiltonian on each basis vector and indentify the tag of the outcoming
    vector, thus we have the value and position of all non-zero elements.

    Tag function: use the incommensurability of prime numbers to make sure
    the tags are unique. Several often used tag functions are:
            T(v) = ∑_{i=1}^M ln(pi) A_{vi}
            T(v) = ∑_{i=1}^M √pi A_{vi}
    where pi is the i-th prime number and A_{vi} is the i-th elements of basis
    vector |v>.
    '''

'''Here we use the tag function: T(v) = ∑_{i=1}^M √pi A_{vi}'''
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

#-------------------------------------------------------------------------------
'''Setup system size and generate basis vectors:
       M: number of sites
       N: number of bosons
       D = (M+N-1)!/ [N!(M-1)!] : Dimension of Hilbert space'''

'''Setup Hamiltonian matrix:
        H_kin : kinetic energy part
            - J ∑_<ij> ai+ aj + h.c.
            J : hopping
        H_int : interaction part
            U/2 ∑_i ni (ni -1)
            U : Onsite interaction'''
def fac(N):
    return np.math.factorial(N)

def space_dim(M,N):
'''Hilbert space dimension'''
    return int(fac(M+N-1)/(fac(N) * fac(M-1)))

def basis_vecs(M,N):
'''Basis vectors generation'''
    D = int(fac(M+N-1)/(fac(N) * fac(M-1)))
    vecs = np.zeros([D,M])
    vecs[0,0] = N

    for i in range(D-1):
        # Determin k
        k = M-2
        for j in range(M-2,-1,-1):
            if vecs[i,j] == 0:
                k = k -1
                if k >= 0:
                    continue
                else:
                    print('Error: zero vector!')
            elif vecs[i,j] != 0:
                break
        # Set the next vector
        vecs[i+1,k] = vecs[i,k] -1

        sum = vecs[i+1,k]
        for n in range(k):
            vecs[i+1,n] = vecs[i,n]
            sum = sum + vecs[i+1,n]

        vecs[i+1,k+1] = N - sum

        for n in range(k+2,M):
            vecs[i+1,n] = 0
    return vecs

def bose_hubbard(M, N, J, U, periodic = False):
    D = space_dim(M,N)
    basis = basis_vecs(M,N)

    T = np.zeros(D)
    for i in range(D):
        T[i] = find_tag(basis[i,:],M)
    Tsorted = sorted(T)
    ind = np.argsort(T)  # Tsorted[i] = T[ind[i]]

    H_int = np.zeros([D,D])
    H_kin = np.zeros([D,D])

    # H_int: diagonal terms
    for i in range(D):
        for j in range(M):
            H_int[i,i] += U/2 * basis[i,j] * (basis[i,j] - 1)

    # H_kin: off-diagonal terms
    for i in range(D):
        if periodic:
            limit = M
        else:
            limit = M-1
        for j in range(limit): # only define the forward jumping
            if basis[i,j] > 0 :
                vec = basis[i,:].copy()
                next = (j+1) % M
                vec[next] += 1
                vec[j] -= 1
                tag = find_tag(vec) # tag for the newly generated vector
                # Find the position of the new vector
                for k in range(M):
                    if tag == Tsorted[k]:
                        next = (j+1) % M
                        H_kin[ind[k],i] += -J * np.sqrt(basis[i,j] * (basis[i,next] + 1))

    H_kin = H_kin + np.conj(H_kin).T
    H = H_int + H_kin
    return H

#-------------------------------------------------------------------------------
'''Some physical quantities'''
def density_matrix(H,basis):
    '''Single particle density matrix'''
    (D,M) = basis.shape
    T = np.zeros(D)
    for i in range(D):
        T[i] = find_tag(basis[i,:],M)
    Tsorted = sorted(T)
    ind = np.argsort(T)  # Tsorted[i] = T[ind[i]]

    Evals, Evecs = sp.sparse.linalg.eigsh(H, k=1, which = 'SA')

    rho = np.zeros([M,M])

    for i in range(M):
        for n in range(D):
            rho[i,i] += basis[n,i] * np.conj(Evecs[n]) * Evecs[n]

    for i in range(M):
        for j in range(M):
            if i != j:
                for n in range(D):
                    if basis[n,j] > 0:
                        vec = basis[n,:].copy()
                        vec[i] += 1
                        vec[j] -= 1
                        tag = find_tag(vec, len(vec))
                        for k in range(D):
                            if tag == Tsorted[k]:
                                rho[i,j] = np.sqrt((basis[n,i]+1) * basis[n,j]) *\
                                            np.conj(Evecs[ind[k]]) * Evecs[j]
    return rho

M = 3
N = 2
J = 1
u = np.linspace(0,20,100)
fc = np.zeros(100)

H = bose_hubbard(M,N,J,0,periodic = True)
Evals, Evecs = sp.sparse.linalg.eigsh(H, k=1, which = 'SA')
basis = basis_vecs(M,N)
print(basis)
print(Evals)
print(Evecs)

'''
i = 0
for U in u:
    H = bose_hubbard(M,N,J,U,periodic = True)
    basis = basis_vecs(M,N)
    rho = density_matrix(H,basis)
    eval = np.linalg.eigvals(rho)
    fc[i] = eval[0] / N
    i += 1

plt.plot(u,fc)
plt.show()
'''
