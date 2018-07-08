"""
File: examples/distribution/sech_squared_distribution.py
Author: Keith Tauscher
Date: 8 Jul 2018

Description: Example of using the SechSquaredDistribution class to represent 1D
             random variates.
"""
import os, time
import numpy as np
import matplotlib.pyplot as pl
from distpy import SechSquaredDistribution

sample_size = int(1e5)
umean = 12.5
uvar = 2.5
distribution = SechSquaredDistribution(umean, uvar)
hdf5_file_name = 'TEST_DELETE_THIS.hdf5'
distribution.save(hdf5_file_name)
try:
    assert distribution == SechSquaredDistribution.load(hdf5_file_name)
except:
    os.remove(hdf5_file_name)
    raise
else:
    os.remove(hdf5_file_name)
assert distribution.numparams == 1
t0 = time.time()
sample = distribution.draw(sample_size)
print(('It took {0:.5f} s for a sample of size {1} to be drawn from a ' +\
    'sech-squared distribution.').format(time.time() - t0, sample_size))
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
pl.title('sech-squared distribution with mean={0!s} and variance={1!s}'.format(\
    umean, uvar), size='xx-large')
pl.xlabel('Value', size='xx-large')
pl.ylabel('PDF', size='xx-large')
pl.tick_params(labelsize='xx-large', width=2, length=6)
pl.legend(fontsize='xx-large')
pl.show()

