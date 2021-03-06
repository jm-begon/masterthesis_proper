# -*- coding: utf-8 -*-
"""
Created on Sun May 04 13:15:06 2014

@author: Jm Begon
"""
import scipy.sparse as sps
import numpy as np
cimport numpy as np
cimport cython


def computeHistogram(obj, nbFactor, nbTrees, A):
    if not sps.isspmatrix_csr:
        A = A.tocsr()

    rowOut = np.zeros((obj*nbFactor*nbTrees), dtype=np.int)
    columnOut = np.zeros((obj*nbFactor*nbTrees), dtype=np.int)
    dataOut = np.zeros((obj*nbFactor*nbTrees), dtype=np.int)
    nbFeatures = A.shape[1]

    size = csr_sum(obj, nbFeatures, nbFactor, nbTrees,
                   A.data, A.indices, A.indptr,
                   rowOut, columnOut, dataOut)

    return sps.coo_matrix(
        (dataOut[:size], (rowOut[:size], columnOut[:size])),
        shape=(obj, nbFeatures))



@cython.wraparound(False)  # Turn off wrapping capabilities (speed up)
@cython.boundscheck(False)  # Turn off bounds-checking for entire function (speed up)
def csr_sum(unsigned int nbObj,
            unsigned int nbFeatures,
            unsigned int nbFactor,
            unsigned int nbTrees,
            np.ndarray[np.float64_t, ndim=1] data,
            np.ndarray[int, ndim=1] indices,
            np.ndarray[int, ndim=1] indptr,
            np.ndarray[np.int_t, ndim=1] rowOut,
            np.ndarray[np.int_t, ndim=1] columnOut,
            np.ndarray[np.int_t, ndim=1] dataOut):

    cdef unsigned int row, col, countVal, j, k, index

    cdef np.ndarray[np.int_t, ndim=1] counter = np.zeros((nbFeatures), dtype=int)
    cdef np.ndarray[np.int_t, ndim=1] colStore = np.zeros((nbTrees*nbFactor), dtype=int)


    k = 0
    for row in xrange(nbObj):
        j = 0
        for index in xrange(indptr[row*nbFactor], indptr[(row+1)*nbFactor]):
            col = indices[index]
            counter[<unsigned int>col] = counter[<unsigned int>col] + 1
            colStore[<unsigned int>j] = col
            j += 1
        for j in xrange(nbTrees*nbFactor):
            col = colStore[<unsigned int>j]
            countVal = counter[<unsigned int>col]
            if countVal != 0:
                rowOut[<unsigned int>k] = row
                columnOut[<unsigned int>k] = col
                dataOut[<unsigned int>k] = countVal
                counter[<unsigned int>col] = 0
                k += 1

    return k
