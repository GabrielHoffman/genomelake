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
        row_idx += 1



cpdef void one_hot_encode_sequence_v3(str seq, float[:, :] encoded) except *:
    cdef size_t row_idx = 0

    if encoded.shape[0] != len(seq):
        raise ValueError('encoded array not the same length as given seq')

    if encoded.shape[1] != 4:
        raise ValueError('encoded array needs to have 4 columns')
       
    hsh = {'A': [(0, 1)], 
    'C': [(1, 1)], 
    'G': [(2, 1)], 
    'T': [(3, 1)], 
    'N': [(0, 0.25), (1, 0.25), (2, 0.25), (3, 0.25)],
    'R': [(0, 0.5), (2, 0.5)], 
    'Y': [(1, 0.5), (3, 0.5)], 
    'S': [(2, 0.5), (1, 0.5)], 
    'W': [(0, 0.5), (3, 0.5)], 
    'K': [(2, 0.5), (3, 0.5)], 
    'M': [(0, 0.5), (1, 0.5)], 
    'B': [(1, 0.3333333333333333), (2, 0.3333333333333333), (3, 0.3333333333333333)], 
    'D': [(0, 0.3333333333333333), (2, 0.3333333333333333), (3, 0.3333333333333333)], 
    'H': [(0, 0.3333333333333333), (1, 0.3333333333333333), (3, 0.3333333333333333)], 
    'V': [(0, 0.3333333333333333), (1, 0.3333333333333333), (2, 0.3333333333333333)]}
    
    cdef int i
    for base in seq:
        for item in hsh[base]:
            i = item[0]
            encoded[row_idx, i] = item[1]
        row_idx += 1

def convertCoding( a ):
  b = np.zeros([4,])
  for entry in a:
    b[entry[0]] = entry[1]
  return( b)

# a = hsh['R']
# convertCoding( a )

def one_hot_encode_sequence_ambig( seq, encoded):

    # Nucleotide definitions
    # ----------------------

    # Standard nucleotides
    # A a 0
    # C c 1
    # G g 2
    # T t 3
    h = { 'A': 0, 'a': 0,
          'C': 1, 'c': 1,
          'G': 2, 'g': 2,
          'T': 3, 'T': 3,
          'U': 3, 'U': 3}

    # Ambiguity codes 
    # https://www.bioinformatics.org/sms/iupac.html
    # N: Any nucleotide (A or C or G or T or U)
    # R: A or G
    # Y: C or T
    # S: G or C
    # W: A or T
    # K: G or T
    # M: A or C
    # B: C or G or T
    # D: A or G or T
    # H: A or C or T
    # V: A or C or G
    # . or -  gap
    hsh = { 'A': [(h['A'], 1)],
            'C': [(h['C'], 1)],
            'G': [(h['G'], 1)],
            'T': [(h['T'], 1)],
            'N': [(h['A'], 0.25), (h['C'], 0.25), (h['G'], 0.25), (h['T'], 0.25)],
            'R': [(h['A'], 0.5), (h['G'], 0.5)],
            'Y': [(h['C'], 0.5), (h['T'], 0.5)],
            'S': [(h['G'], 0.5), (h['C'], 0.5)],
            'W': [(h['A'], 0.5), (h['T'], 0.5)],
            'K': [(h['G'], 0.5), (h['T'], 0.5)],
            'M': [(h['A'], 0.5), (h['C'], 0.5)],
            'B': [(h['C'], 1/3), (h['G'], 1/3), (h['T'], 1/3)],
            'D': [(h['A'], 1/3), (h['G'], 1/3), (h['T'], 1/3)],
            'H': [(h['A'], 1/3), (h['C'], 1/3), (h['T'], 1/3)],
            'V': [(h['A'], 1/3), (h['C'], 1/3), (h['G'], 1/3)]}


    # Evaluating hsh and convertCoding can be expensive since it is 
    # run each call to one_hot_encode_sequence
    # so how to either initialize it once, or make it global to 
    # avoid re-initializing each time
    # to pre-compute hsh_converted and then get sting literal 
    # how do I get hsh_converted to be global, persistent across multiple function calls, static
    # hasattr(myfunc, "counter")
    hsh_converted = {}
    for key in hsh:
      hsh_converted[key] = convertCoding( hsh[key] )

    row_idx = 0
    encoded = np.zeros([len(seq), 4])
    for base in seq.upper():
        encoded[row_idx,] = convertCoding( hsh[base] )
        row_idx = row_idx + 1

    return( encoded )








