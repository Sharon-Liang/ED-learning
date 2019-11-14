# This is a practice of using ED to solve 1D Bose-Hubbard Model
# We use hardcore boson here
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from scipy.sparse.linalg import eigsh

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

#-------------------------------------------------------------------------------
