"""
File: examples/jumping/truncated_gaussian_jumping_distribution.py
Author: Keith Tauscher
Date: 5 Jan 2017

Description: Example of using the TruncatedGaussianJumpingDistribution class.
"""
import os, time
import numpy as np
import matplotlib.pyplot as pl
from distpy import TruncatedGaussianJumpingDistribution,\
    load_jumping_distribution_from_hdf5_file

sample_size = int(1e6)
umean1 = 11.
umean2 = 15.
uvar = 2.5
low = 10
high = 15
distribution = TruncatedGaussianJumpingDistribution(uvar, low=low, high=high)
assert distribution.numparams == 1
try:
    file_name = 'TEMPORARY_TEST_DELETE_THIS_IF_IT_EXISTS.hdf5'
    distribution.save(file_name)
    loaded_distribution = load_jumping_distribution_from_hdf5_file(file_name)
    assert loaded_distribution == distribution
except:
    os.remove(file_name)
    raise
else:
    os.remove(file_name)
t0 = time.time()
sample1 = distribution.draw(umean1, sample_size)
sample2 = distribution.draw(umean2, sample_size)
print(('It took {0:.5f} s for two samples of size {1} to be drawn from a ' +\
    'univariate Gaussian jumping distribution.').format(time.time() - t0,\
    sample_size))
pl.figure()
bins = np.linspace(low, high, 100)
pl.hist(sample1, bins=bins, histtype='step', color='b', linewidth=2,\
    label='sampled', density=True)
pl.hist(sample2, bins=bins, histtype='step', color='b', linewidth=2,\
    label='sampled', density=True)
xs = np.linspace(bins[0], bins[-1], 1000)
y1s = list(map((lambda x : np.exp(distribution.log_value(umean1, x))), xs))
y2s = list(map((lambda x : np.exp(distribution.log_value(umean2, x))), xs))
pl.plot(xs, y1s, linewidth=2, color='r', label='e^(log_prior)')
pl.plot(xs, y2s, linewidth=2, color='r', label='e^(log_prior)')
pl.title(('Truncated Gaussian distributions variance={!s}').format(uvar),\
    size='xx-large')
pl.xlabel('Value', size='xx-large')
pl.ylabel('PDF', size='xx-large')
pl.tick_params(labelsize='xx-large', width=2, length=6)
pl.legend(fontsize='xx-large')
pl.show()

