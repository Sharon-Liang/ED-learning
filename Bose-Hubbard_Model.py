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

vec = basis_vecs(4,4)
print(vec)
