"""
File: examples/distribution/load_distribution.py
Author: Keith Tauscher
Date: 26 Feb 2019

Description: Example script showing how distributions can be anonymously loaded
             using the load_distribution_from_hdf5_file function, even if the
             class of the distribution is unknown.
"""
import os
from distpy import UniformDistribution, load_distribution_from_hdf5_file

distribution = UniformDistribution(-1, 6, metadata='a')
hdf5_file_name = 'TEST_DELETE_THIS.hdf5'
distribution.save(hdf5_file_name)
try:
    assert distribution == load_distribution_from_hdf5_file(hdf5_file_name)
except:
    os.remove(hdf5_file_name)
    raise
else:
    os.remove(hdf5_file_name)

