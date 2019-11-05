# This is a practice of using ED to solve Bose-Hubbard Model
# Follow the instruction of arXiv: 1102.4006v1
import numpy as np


def fac(N):
    return np.math.factorial(N)

'''Setups:
   M: number of sites
   N: number of bosons
   D = (M+N-1)!/ [N!(M-1)!] : Dimension of Hilbert space
   '''
def space_dim(M,N):
    return int(fac(M+N-1)/(fac(N) * fac(M-1)))

'''Basis vectors generation'''
def basis_vecs(M,N):
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

# Here we use the tag function: T(v) = ∑_{i=1}^M √pi A_{vi}
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

def tag_fun(vec, M):
    '''calculate the tag of each vectors'''
    prime = primes(M)
    tag = 0
    for i in range(M):
        tag += np.sqrt(prime[i]) * vec[i]
    return tag

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
def bose_hubbard(M,N,J,U):
    D = space_dim(M,N)
    basis = basis_vecs(M,N)

    T = np.zeros(D)
    for i in range(D):
        T[i] = tag_fun(basis[i,:],M)
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
        for j in range(M-1): # only define the forward jumping
            if basis[i,j] > 0 :
                vec = basis[i,:]
                vec[j+1] += 1
                vec[j] -= 1
                tag = tag_fun(vec,M) # tag for the newly generated vector
                # Find the position of the new vector
                for k in range(M):
                    if tag == Tsorted[k]:
                        H_kin[ind[k],i] += -J * np.sqrt(basis[i,j] * (basis[i,j+1] + 1))
    H_kin = H_kin + np.conj(H_kin).T
    H = H_int + H_kin
    return H
    
