"""
File: examples/distribution/infinite_uniform_distribution.py
Author: Keith Tauscher
Date: 25 Mar 2019

Description: Script showing the use of the InfiniteUniformDistribution class.
"""
import os
import numpy as np
from distpy import InfiniteUniformDistribution,\
    load_distribution_from_hdf5_file

hdf5_file_name = 'TESTINGINFINITEUNIFORMDISTRIBUTIONCLASSDELETETHIS.hdf5'

try:
    distribution =\
        InfiniteUniformDistribution(ndim=3, minima=[None, 0], maxima=[0, None])
except ValueError:
    pass # this is supposed to happen
else:
    raise RuntimeError("A ValueError was supposed to be thrown by the " +\
        "initialized distribution in this try/except structure.")

distribution =\
    InfiniteUniformDistribution(ndim=2, minima=[None, 0], maxima=[0, None])

distribution.save(hdf5_file_name)
try:
    assert(distribution == InfiniteUniformDistribution.load(hdf5_file_name))
    assert(distribution == load_distribution_from_hdf5_file(hdf5_file_name))
except:
    os.remove(hdf5_file_name)
    raise
else:
    os.remove(hdf5_file_name)

assert(distribution.log_value(np.array([-2, 2])) == 0)
assert(distribution.log_value(np.array([2, 2])) == -np.inf)
assert(distribution.log_value(np.array([2, -2])) == -np.inf)
assert(distribution.log_value(np.array([-2, -2])) == -np.inf)

