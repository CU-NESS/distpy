"""
File: examples/distribution/kronecker_delta_distribution.py
Author: Keith Tauscher
Date: 19 Feb 2018

Description: File showing examples of how to use the KroneckerDeltaDistribution
             class, including saving and loading it.
"""
import os
import numpy as np
from distpy import KroneckerDeltaDistribution, load_distribution_from_hdf5_file

hdf5_file_name = 'TEST_DELETE_THIS.hdf5'

distribution = KroneckerDeltaDistribution(1)
assert(distribution.draw() == 1)
distribution.save(hdf5_file_name)
try:
    assert(distribution == load_distribution_from_hdf5_file(hdf5_file_name))
except:
    os.remove(hdf5_file_name)
    raise
else:
    os.remove(hdf5_file_name)

distribution = KroneckerDeltaDistribution([1, 3])
assert(np.all(distribution.draw(100) == np.array([1, 3])[np.newaxis,:]))
distribution.save(hdf5_file_name)
try:
    assert(distribution == load_distribution_from_hdf5_file(hdf5_file_name))
except:
    os.remove(hdf5_file_name)
    raise
else:
    os.remove(hdf5_file_name)

