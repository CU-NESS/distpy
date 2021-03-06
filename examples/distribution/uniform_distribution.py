"""
File: examples/distribution/uniform_distribution.py
Author: Keith Tauscher
Date: Oct 15 2019

Description: Example of using the UniformDistribution.
"""
import os, time
import numpy as np
import matplotlib.pyplot as pl
from distpy import UniformDistribution

low = -27.
high = 19.
sample_size = int(1e5)

distribution = UniformDistribution(high, low, metadata=1)
distribution2 = UniformDistribution(low, high)
hdf5_file_name = 'TEST_DELETE_THIS.hdf5'
distribution.save(hdf5_file_name)
try:
    assert (distribution == UniformDistribution.load(hdf5_file_name))
except:
    os.remove(hdf5_file_name)
    raise
else:
    os.remove(hdf5_file_name)
assert distribution.numparams == 1
assert (distribution.low == distribution2.low) and\
    (distribution.high == distribution2.high)
t0 = time.time()
sample = distribution.draw(sample_size)
print(('It took {0:.5f} s to draw {1} points from a univariate uniform ' +\
    'distribution.').format(time.time() - t0, sample_size))
print('Sample mean was {0:.3g}, while expected mean was {1:.3g}.'.format(\
    np.mean(sample), distribution.mean))
print(('Sample standard deviation was {0:.3g}, while expected standard ' +\
    'deviation was {1:.3g}.').format(np.std(sample),\
    distribution.standard_deviation))
fig = pl.figure()
ax = fig.add_subplot(111)
ax.hist(sample, bins=100, histtype='step', color='b', linewidth=2,\
    density=True, label='sampled')
xs = np.arange(-30., 20., 0.01)
distribution.plot(xs, ax=ax, show=False, linewidth=2, color='r',\
    label='e^(log_value)')
ylim = ax.get_ylim()
for xval in distribution.central_confidence_interval(0.5):
    ax.plot(2 * [xval], ylim, color='k')
ax.set_ylim(ylim)
ax.set_title(('Uniform distribution on [{0!s},{1!s}]').format(\
    distribution.low, distribution.high), size='xx-large')
ax.set_xlabel('Value', size='xx-large')
ax.set_ylabel('PDF', size='xx-large')
ax.tick_params(labelsize='xx-large', width=2, length=6)
ax.legend(fontsize='xx-large', loc='lower center')
pl.show()



