"""
File: examples/distribution/distribution_set_seven_variates.py
Author: Keith Tauscher
Date: 7 Aug 2017

Description: File containing example of a seven-variate DistributionSet.
"""
import os, time
import numpy as np
import matplotlib.pyplot as pl
import matplotlib.cm as cm
from distpy import DistributionSet, GaussianDistribution, BetaDistribution,\
    ParallelepipedDistribution

sample_size = int(1e4)

distribution_set = DistributionSet()
distribution_set.add_distribution(GaussianDistribution(5., 4.), 'a')
distribution_set.add_distribution(BetaDistribution(18., 3.), 'b')
distribution_set.add_distribution(\
    GaussianDistribution([5., 2., 9.],\
                         [[4., 1., 1.], [1., 4., 1.], [1., 1., 4.]]),\
                         ['c', 'd', 'e'])
distribution_set.add_distribution(ParallelepipedDistribution([69., 156.],\
                                  [[1.,-1.], [1.,1.]], [1., 1.]),\
                                  ['f', 'g'])
hdf5_file_name = 'TEST_DELETE_THIS.hdf5'
distribution_set.save(hdf5_file_name)
try:
    assert distribution_set == DistributionSet.load(hdf5_file_name)
except:
    os.remove(hdf5_file_name)
    raise
else:
    os.remove(hdf5_file_name)
t0 = time.time()
sample = distribution_set.draw(sample_size)
print(('It took {0:.5f} s to draw {1} points from a mixed distribution ' +\
    'with 7 parameters, in groups of 3, 2, 1, and 1.').format(\
    time.time() - t0, sample_size))
pl.figure()
pl.hist2d(sample['b'], sample['g'], bins=100, cmap=cm.bone)
pl.title('PriorSet seven-parameter test. b should be in [0,1] and g should ' +\
         'be around 156', size='xx-large')
pl.xlabel('b', size='xx-large')
pl.ylabel('g', size='xx-large')
pl.tick_params(labelsize='xx-large', width=2, length=6)
pl.show()

