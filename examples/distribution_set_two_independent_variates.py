"""
File: examples/distribution_set_two_independent_variates.py
Author: Keith Tauscher
Date: 7 Aug 2017

Description: File containing example of how to add two independent random
             variates into the same DistributionSet.
"""
import time
import numpy as np
import matplotlib.pyplot as pl
import matplotlib.cm as cm
from distpy import DistributionSet, UniformDistribution, GaussianDistribution

sample_size = int(1e4)

distribution_set = DistributionSet()
distribution_set.add_distribution(UniformDistribution(-3., 7.), 'x')
distribution_set.add_distribution(GaussianDistribution(5., 9.), 'y')
t0 = time.time()
sample = [distribution_set.draw() for i in range(sample_size)]
print ('It took %.3f s to draw %i' % (time.time()-t0,sample_size,)) +\
      ' points from a 2 parameter pdf made up of a ' +\
      'uniform distribution times a Gaussian.'
xs = [sample[i]['x'] for i in range(sample_size)]
ys = [sample[i]['y'] for i in range(sample_size)]
pl.figure()
pl.hist2d(xs, ys, bins=100, cmap=cm.bone)
pl.title("PriorSet 2 independent parameter test. x is Unif(-3, 7) and y is " +\
         "Gaussian(5., 9.)", size='xx-large')
pl.xlabel('x', size='xx-large')
pl.ylabel('y', size='xx-large')
pl.tick_params(labelsize='xx-large', width=2, length=6)
pl.show()

