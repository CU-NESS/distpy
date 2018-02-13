"""
File: examples/distribution/beta_distribution.py
Author: Keith Tauscher
Date: 7 Aug 2017

Description: Example of using the BetaDistribution class.
"""
import os, time
import numpy as np
import matplotlib.pyplot as pl
from distpy import BetaDistribution

sample_size = int(1e5)

distribution = BetaDistribution(9, 1, metadata='a')
hdf5_file_name = 'TEST_DELETE_THIS.hdf5'
distribution.save(hdf5_file_name)
try:
    assert distribution == BetaDistribution.load(hdf5_file_name)
except:
    os.remove(hdf5_file_name)
    raise
else:
    os.remove(hdf5_file_name)
assert distribution.numparams == 1
t0 = time.time()
sample = distribution.draw(sample_size)
print('It took {0:.5f} s to draw {1} points from a beta distribution.'.format(\
    time.time() - t0, sample_size))
pl.figure()
pl.hist(sample, bins=100, linewidth=2, color='b', histtype='step',\
    normed=True, label='sampled')
xs = np.arange(0.3, 1.0, 0.001)
pl.plot(xs, list(map((lambda x : np.exp(distribution.log_value(x))), xs)),\
    linewidth=2, color='r', label='e^(log_value)')
ylim = pl.ylim()
for xval in distribution.right_confidence_interval(0.5):
    pl.plot(2 * [xval], ylim, color='k')
pl.ylim(ylim)
pl.title('Beta distribution test', size='xx-large')
pl.xlabel('Value', size='xx-large')
pl.ylabel('PDF', size='xx-large')
pl.tick_params(labelsize='xx-large', width=2, length=6)
pl.legend(fontsize='xx-large', loc='upper left')
pl.show()

