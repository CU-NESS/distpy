"""
File: examples/distribution_set_two_independent_variates.py
Author: Keith Tauscher
Date: 7 Aug 2017

Description: File containing example of how to add two distributions containing
             information about three different random variates.
"""
import time
import numpy as np
import matplotlib.pyplot as pl
import matplotlib.cm as cm
from distpy import DistributionSet, UniformDistribution, GaussianDistribution

sample_size = int(1e4)

distribution_set = DistributionSet()
distribution_set.add_distribution(\
    GaussianDistribution([10., -5.], [[2., 1.], [1., 2.]]), ['x', 'y'])
distribution_set.add_distribution(UniformDistribution(-3., 17.), 'z')
t0 = time.time()
sample = distribution_set.draw(sample_size)
print ('It took %.3f s to draw %i' % (time.time()-t0,sample_size,)) +\
       ' points from a 3 parameter pdf made up of a 2D ' +\
       'gaussian and a 1D uniform distribution'
pl.figure()
pl.hist2d(sample['x'], sample['z'], bins=100, cmap=cm.bone)
pl.title('PriorSet 3 parameters with correlation between x and z',\
    size='xx-large')
pl.xlabel('x', size='xx-large')
pl.ylabel('z', size='xx-large')
pl.tick_params(labelsize='xx-large', width=2, length=6)
pl.show()

