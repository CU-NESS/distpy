"""
File: examples/distribution/weibull_distribution.py
Author: Keith Tauscher
Date: 7 Aug 2017

Description: Example of using the WeibullDistribution class.
"""
import time
import numpy as np
import matplotlib.pyplot as pl
from distpy import WeibullDistribution

sample_size = int(1e5)

distribution = WeibullDistribution(5)
assert distribution.numparams == 1
t0 = time.time()
sample = distribution.draw(sample_size)
print(('It took {0:.5f} s to draw {1} points from a Weibull ' +\
    'distribution.').format(time.time() - t0, sample_size))
pl.figure()
pl.hist(sample, bins=100, linewidth=2, color='b', histtype='step',\
    label='sampled', normed=True)
xs = np.arange(0.001, 3., 0.001)
pl.plot(xs, list(map((lambda x : np.exp(distribution.log_value(x))), xs)),\
    linewidth=2, color='r', label='e^(log_value)')
ylim = pl.ylim()
for xval in distribution.central_confidence_interval(0.5):
    pl.plot(2 * [xval], ylim, color='k')
pl.ylim(ylim)
pl.title('Weibull distribution test', size='xx-large')
pl.xlabel('Value', size='xx-large')
pl.ylabel('PDF', size='xx-large')
pl.tick_params(labelsize='xx-large', width=2, length=6)
pl.legend(fontsize='xx-large')
pl.show()

