"""
File: examples/distribution/double_sided_exponential_distribution.py
Author: Keith Tauscher
Date: Oct 15 2019

Description: Example of using the DoubleSidedExponentialDistribution class.
"""
import os, time
import numpy as np
import matplotlib.pyplot as pl
from distpy import DoubleSidedExponentialDistribution

sample_size = int(1e5)

distribution = DoubleSidedExponentialDistribution(0., 1.)
assert distribution.numparams == 1
hdf5_file_name = 'TEST_DELETE_THIS.hdf5'
distribution.save(hdf5_file_name)
try:
    assert\
        distribution == DoubleSidedExponentialDistribution.load(hdf5_file_name)
except:
    os.remove(hdf5_file_name)
    raise
else:
    os.remove(hdf5_file_name)
t0 = time.time()
sample = distribution.draw(sample_size)
print(('It took {0:.5f} s to draw {1} points from a double-sided ' +\
    'exponential distribution.').format(time.time() - t0, sample_size))
print('Sample mean was {0:.3g}, while expected mean was {1:.3g}.'.format(\
    np.mean(sample), distribution.mean))
print(('Sample standard deviation was {0:.3g}, while expected standard ' +\
    'deviation was {1:.3g}.').format(np.std(sample),\
    distribution.standard_deviation))
fig = pl.figure()
ax = fig.add_subplot(111)
ax.hist(sample, bins=100, histtype='step', color='b', linewidth=2,\
    density=True, label='sampled')
xs = np.arange(-9., 9., 0.01)
distribution.plot(xs, ax=ax, show=False, color='r', linewidth=2,\
    label='e^(log_value)')
ylim = ax.get_ylim()
for xval in distribution.central_confidence_interval(0.5):
    ax.plot(2 * [xval], ylim, color='k')
ax.set_ylim(ylim)
ax.legend(fontsize='xx-large', loc='upper right')
ax.set_title('Double-sided exponential distribution test', size='xx-large')
ax.set_xlabel('Value', size='xx-large')
ax.set_ylabel('PDF', size='xx-large')
ax.tick_params(labelsize='xx-large', width=2, length=6)
pl.show()

