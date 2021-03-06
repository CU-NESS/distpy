"""
File: examples/distribution/univariate_gaussian_distribution.py
Author: Keith Tauscher
Date: Oct 15 2019

Description: Example of using the GaussianDistribution class to represent 1D
             Gaussian random variates.
"""
import os, time
import numpy as np
import matplotlib.pyplot as pl
from distpy import GaussianDistribution

sample_size = int(1e5)
umean = 12.5
uvar = 2.5
distribution = GaussianDistribution(umean, uvar)
hdf5_file_name = 'TEST_DELETE_THIS.hdf5'
distribution.save(hdf5_file_name)
try:
    assert distribution == GaussianDistribution.load(hdf5_file_name)
except:
    os.remove(hdf5_file_name)
    raise
else:
    os.remove(hdf5_file_name)
assert distribution.numparams == 1
t0 = time.time()
sample = distribution.draw(sample_size)
print(('It took {0:.5f} s for a sample of size {1} to be drawn from a ' +\
    'univariate Gaussian.').format(time.time() - t0, sample_size))
print('Sample mean was {0:.3g}, while expected mean was {1:.3g}.'.format(\
    np.mean(sample), distribution.mean))
print(('Sample standard deviation was {0:.3g}, while expected standard ' +\
    'deviation was {1:.3g}.').format(np.std(sample),\
    distribution.standard_deviation))
fig = pl.figure()
ax = fig.add_subplot(111)
ax.hist(sample, bins=100, histtype='step', color='b', linewidth=2,\
    label='sampled', density=True)
xs = np.arange(5., 20., 0.01)
distribution.plot(xs, ax=ax, show=False, linewidth=2, color='r',\
    label='e^(log_prior)')
ylim = ax.get_ylim()
for xval in distribution.central_confidence_interval(0.6827):
    ax.plot(2 * [xval], ylim, color='k')
ax.set_ylim(ylim)
ax.set_title(('Univariate Gaussian distribution with mean={0!s} and ' +\
    'variance={1!s}').format(umean, uvar), size='xx-large')
ax.set_xlabel('Value', size='xx-large')
ax.set_ylabel('PDF', size='xx-large')
ax.tick_params(labelsize='xx-large', width=2, length=6)
ax.legend(fontsize='xx-large')
pl.show()

