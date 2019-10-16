"""
File: examples/distribution/distribution_set_two_independent_variates.py
Author: Keith Tauscher
Date: 7 Aug 2017

Description: File containing example of how to add two distributions containing
             information about three different random variates.
"""
import os, time
import numpy as np
import matplotlib.pyplot as pl
import matplotlib.cm as cm
from distpy import DistributionSet, UniformDistribution, GaussianDistribution

sample_size = int(1e4)

distribution_set = DistributionSet()
distribution_set.add_distribution(\
    GaussianDistribution([10., -5.], [[2., 1.], [1., 2.]]), ['x', 'y'])
distribution_set.add_distribution(UniformDistribution(-3., 17.), 'z')

hdf5_file_name = 'TEST_DELETE_THIS.hdf5'
distribution_set.save(hdf5_file_name)
try:
    assert(distribution_set == DistributionSet.load(hdf5_file_name))
    assert(distribution_set.bounds['x'] == (None, None))
    assert(distribution_set.bounds['y'] == (None, None))
    assert(distribution_set.bounds['z'] == (-3, 17))
except:
    os.remove(hdf5_file_name)
    raise
else:
    os.remove(hdf5_file_name)

t0 = time.time()
sample = distribution_set.draw(sample_size)
print(('It took {0:.5f} s to draw {1} points from a 3-parameter pdf made ' +\
    'up of a 2D Gaussian and a 1D uniform distribution.').format(\
    time.time() - t0, sample_size))
print('sample_mean={}, expected_mean={}'.format(\
    {key: np.mean(sample[key]) for key in sample}, distribution_set.mean))
print('sample_variances={}, expected_variances={}'.format(\
    {key: np.var(sample[key]) for key in sample}, distribution_set.variance))
pl.figure()
pl.hist2d(sample['x'], sample['z'], bins=100, cmap=cm.bone)
pl.title('PriorSet 3 parameters with correlation between x and z',\
    size='xx-large')
pl.xlabel('x', size='xx-large')
pl.ylabel('z', size='xx-large')
pl.tick_params(labelsize='xx-large', width=2, length=6)
pl.show()

