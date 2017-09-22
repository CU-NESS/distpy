"""
File: examples/distribution/univariate_gaussian_distribution.py
Author: Keith Tauscher
Date: 7 Aug 2017

Description: Example of using the GaussianDistribution class to represent 1D
             Gaussian random variates.
"""
import time
import numpy as np
import matplotlib.pyplot as pl
from distpy import GaussianDistribution

sample_size = int(1e5)
umean = 12.5
uvar = 2.5
distribution = GaussianDistribution(umean, uvar)
assert distribution.numparams == 1
t0 = time.time()
sample = distribution.draw(sample_size)
print(('It took {0:.5f} s for a sample of size {1} to be drawn from a ' +\
    'univariate Gaussian.').format(time.time() - t0, sample_size))
pl.figure()
pl.hist(sample, bins=100, histtype='step', color='b', linewidth=2,\
    label='sampled', normed=True)
xs = np.arange(5., 20., 0.01)
pl.plot(xs, list(map((lambda x : np.exp(distribution.log_value(x))), xs)),\
    linewidth=2, color='r', label='e^(log_prior)')
ylim = pl.ylim()
for xval in distribution.central_confidence_interval(0.6827):
    pl.plot(2 * [xval], ylim, color='k')
pl.ylim(ylim)
pl.title(('Univariate Gaussian distribution with mean={0!s} and ' +\
    'variance={1!s}').format(umean, uvar), size='xx-large')
pl.xlabel('Value', size='xx-large')
pl.ylabel('PDF', size='xx-large')
pl.tick_params(labelsize='xx-large', width=2, length=6)
pl.legend(fontsize='xx-large')
pl.show()

