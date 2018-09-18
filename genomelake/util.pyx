# cython: embedsignature=True
import os
import sys
import numpy as np

from libc.math cimport isnan


def makedirs(path, mode=0777, exist_ok=False):
    try:
        os.makedirs(path, mode)
    except OSError:
        if not exist_ok or not os.path.isdir(path):
            raise
        else:
            sys.stderr.write('Warning: directory %s exists.\n' % path)


cdef inline int char2index(char ch) except -2:
    if ch == 'A' or ch == 'a':
        return 0
    if ch == 'C' or ch == 'c':
        return 1
    if ch == 'G' or ch == 'g':
        return 2
    if ch == 'T' or ch == 't':
        return 3
    if ch == 'N' or ch == 'n':
        return -1
    raise ValueError('Invalid base encountered.')


cpdef void one_hot_encode_sequence(str seq, float[:, :] encoded) except *:
    cdef size_t row_idx = 0
    cdef int col_idx

    if encoded.shape[0] != len(seq):
        raise ValueError('encoded array not the same length as given seq')

    if encoded.shape[1] != 4:
        raise ValueError('encoded array needs to have 4 columns')
    
    for base in seq:
        col_idx = char2index(ord(base))

        if col_idx >= 0:
            encoded[row_idx, col_idx] = 1
        else:
            encoded[row_idx, :] = 0.25

        row_idx += 1


cpdef nan_to_zero(float[:] arr):
    cdef Py_ssize_t k
    for k in range(arr.size):
        if isnan(arr[k]):
            arr[k] = 0


cpdef void one_hot_encode_sequence_v2(str seq, float[:, :] encoded) except *:
    cdef size_t row_idx = 0

    if encoded.shape[0] != len(seq):
        raise ValueError('encoded array not the same length as given seq')

    if encoded.shape[1] != 4:
        raise ValueError('encoded array needs to have 4 columns')
    
    hsh = {
    'A': [1., 0., 0., 0.], 
    'C': [0., 1., 0., 0.], 
    'G': [0., 0., 1., 0.], 
    'T': [0., 0., 0., 1.], 
    'N': [0.25, 0.25, 0.25, 0.25],
    'R': [0.5, 0. , 0.5, 0. ], 
    'Y': [0. , 0.5, 0. , 0.5], 
    'S': [0. , 0.5, 0.5, 0. ], 
    'W': [0.5, 0. , 0. , 0.5], 
    'K': [0. , 0. , 0.5, 0.5], 
    'M': [0.5, 0.5, 0. , 0. ], 
    'B': [0.        , 0.33333333, 0.33333333, 0.33333333], 
    'D': [0.33333333, 0.        , 0.33333333, 0.33333333], 
    'H': [0.33333333, 0.33333333, 0.        , 0.33333333], 
    'V': [0.33333333, 0.33333333, 0.33333333, 0.        ]}
   
    for base in seq:
        value = hsh[base]
        for i in range(0,4):
            encoded[row_idx,i] = value[i]
        # encoded[row_idx, ] = np.array([2.,2.,2.,2.])
        row_idx += 1