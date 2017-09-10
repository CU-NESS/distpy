"""
File: examples/distribution_set_square_transform.py
Author: Keith Tauscher
Date: 7 Aug 2017

Description: File containing script which demonstrates the effectiveness of the
             'square' transform in the DistributionSet class. In this example,
             the DistributionSet has only one Distribution.
"""
import time
import numpy as np
import matplotlib.pyplot as pl
from distpy import DistributionSet, UniformDistribution

sample_size = int(1e4)

distribution_set = DistributionSet()
distribution_set.add_distribution(UniformDistribution(1., 90.), 'x', 'square')
t0 = time.time()
sample = distribution_set.draw(sample_size)['x']
print(('It took {0:.5f} s to draw {1} points from a 1D uniform ' +\
    'distribution (in square space).').format(time.time() - t0, sample_size))
pl.figure()
pl.hist(sample, bins=100, histtype='step', color='b', linewidth=2,\
    label='sampled', normed=True)
xs = np.arange(0.001, 10, 0.001)
pl.plot(xs,\
    list(map(lambda x : np.exp(distribution_set.log_value({'x': x})), xs)),\
    linewidth=2, color='r', label='e^(log_value)')
pl.legend(fontsize='xx-large', loc='upper left')
pl.title('Uniform distribution in square space', size='xx-large')
pl.xlabel('Value', size='xx-large')
pl.ylabel('PDF', size='xx-large')
pl.tick_params(labelsize='xx-large', width=2, length=6)
pl.show()

