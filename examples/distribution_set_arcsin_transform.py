"""
File: examples/distribution_set_arcsin_transform.py
Author: Keith Tauscher
Date: 7 Aug 2017

Description: File containing script which demonstrates the effectiveness of the
             'arcsin' transform in the DistributionSet class. In this example,
             the DistributionSet has only one Distribution.
"""
import time
import numpy as np
import matplotlib.pyplot as pl
from distpy import DistributionSet, UniformDistribution

sample_size = int(1e4)

distribution_set = DistributionSet()
distribution_set.add_distribution(UniformDistribution(0, np.pi / 2.), 'x',\
    'arcsin')
t0 = time.time()
sample = distribution_set.draw(sample_size)['x']
print ('It took %.3f s to draw %i ' % (time.time()-t0,sample_size,)) +\
      'points from a 1D uniform distribution (in arcsin space).'
pl.figure()
pl.hist(sample, bins=100, histtype='step', color='b', linewidth=2,\
    label='sampled', normed=True)
xs = np.linspace(0.001, 0.999, 999)
pl.plot(xs, map(lambda x : np.exp(distribution_set.log_value({'x': x})), xs),\
    linewidth=2, color='r', label='e^(log_value)')
pl.legend(fontsize='xx-large', loc='upper left')
pl.title('Uniform distribution in arcsin space', size='xx-large')
pl.xlabel('Value', size='xx-large')
pl.ylabel('PDF', size='xx-large')
pl.tick_params(labelsize='xx-large', width=2, length=6)
pl.show()

