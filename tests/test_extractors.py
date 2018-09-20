from genomelake import backend
from genomelake.extractors import ArrayExtractor, BigwigExtractor, FastaExtractor
import numpy as np
from pybedtools import Interval
import pyBigWig
import pytest

array_extractor_fasta_params = [("numpy", True),
                                ("numpy", False),
                                ("bcolz", True),
                                ("bcolz", False)]

def test_fasta_extractor_valid_intervals():
    extractor = FastaExtractor('tests/data/fasta_test.fa')
    intervals = [Interval('chr1', 0, 10),
                 Interval('chr2', 0, 10)]
    expected_data = np.array(
        [[[ 1.  ,  0.  ,  0.  ,  0.  ],
          [ 0.  ,  1.  ,  0.  ,  0.  ],
          [ 0.  ,  1.  ,  0.  ,  0.  ],
          [ 0.  ,  0.  ,  1.  ,  0.  ],
          [ 0.  ,  0.  ,  0.  ,  1.  ],
          [ 1.  ,  0.  ,  0.  ,  0.  ],
          [ 0.  ,  1.  ,  0.  ,  0.  ],
          [ 0.  ,  1.  ,  0.  ,  0.  ],
          [ 0.  ,  0.  ,  1.  ,  0.  ],
          [ 0.  ,  0.  ,  0.  ,  1.  ]],

         [[ 1.  ,  0.  ,  0.  ,  0.  ],
          [ 0.  ,  1.  ,  0.  ,  0.  ],
          [ 0.  ,  0.  ,  1.  ,  0.  ],
          [ 0.  ,  0.  ,  0.  ,  1.  ],
          [ 0.25,  0.25,  0.25,  0.25],
          [ 1.  ,  0.  ,  0.  ,  0.  ],
          [ 0.  ,  1.  ,  0.  ,  0.  ],
          [ 0.  ,  0.  ,  1.  ,  0.  ],
          [ 0.  ,  0.  ,  0.  ,  1.  ],
          [ 0.25,  0.25,  0.25,  0.25]]], dtype=np.float32)
    data = extractor(intervals)
    assert (data == expected_data).all()


def test_fasta_extractor_over_chr_end():
    extractor = FastaExtractor('tests/data/fasta_test.fa')
    intervals = [Interval('chr1', 0, 100),
                 Interval('chr1', 1, 101)]
    with pytest.raises(ValueError):
        data = extractor(intervals)

@pytest.mark.parametrize("mode,in_memory", array_extractor_fasta_params)
def test_array_extractor_fasta(mode, in_memory):
    data_dir = 'tests/data/fasta_test_dir_{}_{}'.format(mode, in_memory)
    backend.extract_fasta_to_file(
        'tests/data/fasta_test.fa',
        data_dir,
        mode=mode,
        overwrite=True)
    extractor = ArrayExtractor(data_dir, in_memory=in_memory)
    intervals = [Interval('chr1', 0, 10),
                 Interval('chr2', 0, 10)]
    expected_data = np.array(
        [[[ 1.  ,  0.  ,  0.  ,  0.  ],
          [ 0.  ,  1.  ,  0.  ,  0.  ],
          [ 0.  ,  1.  ,  0.  ,  0.  ],
          [ 0.  ,  0.  ,  1.  ,  0.  ],
          [ 0.  ,  0.  ,  0.  ,  1.  ],
          [ 1.  ,  0.  ,  0.  ,  0.  ],
          [ 0.  ,  1.  ,  0.  ,  0.  ],
          [ 0.  ,  1.  ,  0.  ,  0.  ],
          [ 0.  ,  0.  ,  1.  ,  0.  ],
          [ 0.  ,  0.  ,  0.  ,  1.  ]],

         [[ 1.  ,  0.  ,  0.  ,  0.  ],
          [ 0.  ,  1.  ,  0.  ,  0.  ],
          [ 0.  ,  0.  ,  1.  ,  0.  ],
          [ 0.  ,  0.  ,  0.  ,  1.  ],
          [ 0.25,  0.25,  0.25,  0.25],
          [ 1.  ,  0.  ,  0.  ,  0.  ],
          [ 0.  ,  1.  ,  0.  ,  0.  ],
          [ 0.  ,  0.  ,  1.  ,  0.  ],
          [ 0.  ,  0.  ,  0.  ,  1.  ],
          [ 0.25,  0.25,  0.25,  0.25]]], dtype=np.float32)
    data = extractor(intervals)
    assert (data == expected_data).all()

@pytest.fixture
def test_bigwig_and_intervals():
    bw_path = "tests/data/test_bigwig.bw"
    intervals = [Interval('chr1', 0, 10),
                 Interval('chr2', 0, 10)]
    expected_chr1 = np.array([0.1] * 10, dtype=np.float32)
    expected_chr2 = np.array([0] + [9]*9, dtype=np.float32)
    expected_data = np.stack([expected_chr1, expected_chr2])

    return (bw_path, intervals, expected_data)

@pytest.mark.parametrize("mode,in_memory", array_extractor_fasta_params)
def test_array_extractor_bigwig(test_bigwig_and_intervals, mode, in_memory):
    bw_path, intervals, expected_data = test_bigwig_and_intervals
    bw_dir_path = "{}.dir".format(bw_path)
    backend.extract_bigwig_to_file(
        bw_path, bw_dir_path, mode=mode, overwrite=True)
    extractor = ArrayExtractor(bw_dir_path, in_memory=in_memory)

    data = extractor(intervals)
    assert (data == expected_data).all()


def test_bigwig_extractor(test_bigwig_and_intervals):
    bw_path, intervals, expected_data = test_bigwig_and_intervals
    extractor = BigwigExtractor(bw_path)
    data = extractor(intervals)
    assert (data == expected_data).all()

# Gabriel Hoffman
# add test functions
from genomelake import util
from genomelake.util import one_hot_encode_sequence
from genomelake.util import one_hot_encode_sequence_v2
from genomelake.util import one_hot_encode_sequence_v3
from genomelake.util import one_hot_encode_sequence_ambig
import numpy as np

import cProfile
import timeit

# test that different encoding functions give same results
def test_nucleotide_encoding():
  # set sequence 
  seq = 'ATCGatcgN'
  # initialize numpy arrays
  encode1 = np.zeros([len(seq),4], "float32")
  encode2 = np.zeros([len(seq),4], "float32")
  encode3 = np.zeros([len(seq),4], "float32")
  encode4 = np.zeros([len(seq),4], "float32")
  # run encoding functions
  one_hot_encode_sequence(seq.upper(), encode1[:,:])
  one_hot_encode_sequence_v2(seq.upper(), encode2[:,:])
  one_hot_encode_sequence_v3(seq.upper(), encode3[:,:])
  encode4 = one_hot_encode_sequence_ambig(seq.upper(), encode4[:,:])
  # test 
  assert np.array_equal(encode1, encode2) == True
  assert np.array_equal(encode1, encode3) == True
  assert np.array_equal(encode1, encode4) == True

# test timings
# Profile.run is not working with seq variables 
def test_nucleotide_encoding_runtime():
  # set sequence 
  seq = 'ATCGatcgN'
  for i in range(0, 1):
    seq = seq + seq
  # print(seq)
  # initialize numpy arrays
  encode1 = np.zeros([len(seq),4], "float32")
  encode2 = np.zeros([len(seq),4], "float32")
  encode3 = np.zeros([len(seq),4], "float32")
  encode4 = np.zeros([len(seq),4], "float32")
  # timings
  # cProfile.run("[one_hot_encode_sequence(seq.upper(), encode1[:,:]) for x in range(10000)]")
  # cProfile.run("[one_hot_encode_sequence_v2(seq.upper(), encode2[:,:]) for x in range(10000)]")
    # cProfile.run("[one_hot_encode_sequence_v3(seq.upper(), encode3[:,:]) for x in range(10000)]")
  # cProfile.run("[one_hot_encode_sequence_ambig(seq.upper(), encode4[:,:]) for x in range(10000)]")


test_nucleotide_encoding()

test_nucleotide_encoding_runtime()



