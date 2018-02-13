"""
File: examples/distribution/distribution_set_log_transform.py
Author: Keith Tauscher
Date: 7 Aug 2017

Description: File containing script which demonstrates the effectiveness of the
             'log' transform in the DistributionSet class. In this example the
             DistributionSet has only one Distribution.
"""
import os, time
import numpy as np
import matplotlib.pyplot as pl
from distpy import DistributionSet, GaussianDistribution

sample_size = int(1e4)

distribution_set = DistributionSet()
distribution_set.add_distribution(GaussianDistribution(5., 1.), 'x', 'log')

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
sample = distribution_set.draw(sample_size)['x']
print(('It took {0:.5f} s to draw {1} points from a 1-parameter lognormal ' +\
    'distribution.').format(time.time() - t0, sample_size))
pl.figure()
pl.hist(sample, bins=np.arange(0., 1501., 15.), histtype='step',\
    color='b', linewidth=2, label='sampled', normed=True)
xs = np.arange(0.1, 1500., 0.1)
pl.plot(xs,\
    list(map(lambda x : np.exp(distribution_set.log_value({'x': x})), xs)),\
    linewidth=2, color='r', label='e^(log_value)')
pl.title('Normal distribution in log space', size='xx-large')
pl.xlabel('Value', size='xx-large')
pl.ylabel('PDF', size='xx-large')
pl.legend(fontsize='xx-large', loc='upper right')
pl.tick_params(labelsize='xx-large', width=2, length=6)
pl.show()

