"""
File: examples/distribution/weibull_distribution.py
Author: Keith Tauscher
Date: 7 Aug 2017

Description: Example of using the WeibullDistribution class.
"""
import os, time
import numpy as np
import matplotlib.pyplot as pl
from distpy import WeibullDistribution

sample_size = int(1e5)

distribution = WeibullDistribution(5)
hdf5_file_name = 'TEST_DELETE_THIS.hdf5'
distribution.save(hdf5_file_name)
try:
    assert distribution == WeibullDistribution.load(hdf5_file_name)
except:
    os.remove(hdf5_file_name)
    raise
else:
    os.remove(hdf5_file_name)
assert distribution.numparams == 1
t0 = time.time()
sample = distribution.draw(sample_size)
print(('It took {0:.5f} s to draw {1} points from a Weibull ' +\
    'distribution.').format(time.time() - t0, sample_size))
fig = pl.figure()
ax = fig.add_subplot(111)
ax.hist(sample, bins=100, linewidth=2, color='b', histtype='step',\
    label='sampled', density=True)
xs = np.arange(0.001, 3., 0.001)
distribution.plot(xs, ax=ax, show=False, linewidth=2, color='r',\
    label='e^(log_value)')
ylim = ax.get_ylim()
for xval in distribution.central_confidence_interval(0.5):
    ax.plot(2 * [xval], ylim, color='k')
ax.set_ylim(ylim)
ax.set_title('Weibull distribution test', size='xx-large')
ax.set_xlabel('Value', size='xx-large')
ax.set_ylabel('PDF', size='xx-large')
ax.tick_params(labelsize='xx-large', width=2, length=6)
ax.legend(fontsize='xx-large')
pl.show()

