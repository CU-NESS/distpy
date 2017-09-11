"""
File: examples/distribution/exponential_distribution.py
Author: Keith Tauscher
Date: 7 Aug 2017

Description: Example of using the ExponentialDistribution class.
"""
import time
import numpy as np
import matplotlib.pyplot as pl
from distpy import ExponentialDistribution

sample_size = int(1e5)

distribution = ExponentialDistribution(0.1, shift=-5.)
assert distribution.numparams == 1
t0 = time.time()
sample = distribution.draw(sample_size)
print(('It took {0:.5f} s to draw {1} points from an exponential ' +\
    'distribution.').format(time.time() - t0, sample_size))
pl.figure()
pl.hist(sample, bins=100, histtype='step', color='b', linewidth=2,\
    normed=True, label='sampled')
xs = np.arange(-10., 75., 0.01)
pl.plot(xs, list(map((lambda x : np.exp(distribution.log_value(x))), xs)),\
    linewidth=2, color='r', label='e^(log_value)')
pl.legend(fontsize='xx-large', loc='upper right')
pl.title('Exponential distribution test (mean=5 and shift=-5)',\
    size='xx-large')
pl.xlabel('Value', size='xx-large')
pl.ylabel('PDF', size='xx-large')
pl.tick_params(labelsize='xx-large', width=2, length=6)
pl.show()

