"""
File: examples/distribution/gamma_distribution.py
Author: Keith Tauscher
Date: 7 Aug 2017

Description: Example of using the GammaDistribution class.
"""
import time
import numpy as np
import matplotlib.pyplot as pl
from distpy import GammaDistribution

sample_size = int(1e5)

distribution = GammaDistribution(4, 1)
assert distribution.numparams == 1
t0 = time.time()
sample = distribution.draw(sample_size)
print(('It took {0:.5f} s to draw {1} points from a gamma ' +\
    'distribution.').format(time.time() - t0, sample_size))
pl.figure()
pl.hist(sample, bins=100, linewidth=2, color='b', histtype='step',\
    label='sampled', normed=True)
xs = np.arange(0.001, 18., 0.001)
pl.plot(xs, list(map((lambda x : np.exp(distribution.log_value(x))), xs)),\
    linewidth=2, color='r', label='e^(log_value)')
pl.title('Gamma distribution test', size='xx-large')
pl.xlabel('Value', size='xx-large')
pl.ylabel('PDF', size='xx-large')
pl.tick_params(labelsize='xx-large', width=2, length=6)
pl.legend(fontsize='xx-large')
pl.show()

