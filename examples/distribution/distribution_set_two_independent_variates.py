"""
File: examples/distribution/distribution_set_two_independent_variates.py
Author: Keith Tauscher
Date: 7 Aug 2017

Description: File containing example of how to add two independent random
             variates into the same DistributionSet.
"""
import os, time
import numpy as np
import matplotlib.pyplot as pl
import matplotlib.cm as cm
from distpy import DistributionSet, UniformDistribution, GaussianDistribution

sample_size = int(1e4)

distribution_set = DistributionSet()
distribution_set.add_distribution(UniformDistribution(-3., 7.), 'x')
distribution_set.add_distribution(GaussianDistribution(5., 9.), 'y')

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
print(('It took {0:.5f} s to draw {1} points from a 2-parameter pdf made ' +\
    'up of a uniform distribution times a Gaussian.').format(time.time() - t0,\
    sample_size))
pl.figure()
pl.hist2d(sample['x'], sample['y'], bins=100, cmap=cm.bone)
pl.title("PriorSet 2 independent parameter test. x is Unif(-3, 7) and y is " +\
         "Gaussian(5., 9.)", size='xx-large')
pl.xlabel('x', size='xx-large')
pl.ylabel('y', size='xx-large')
pl.tick_params(labelsize='xx-large', width=2, length=6)
pl.show()

