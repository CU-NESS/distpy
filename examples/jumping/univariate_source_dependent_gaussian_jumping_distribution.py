"""
File: examples/jumping/\
      univariate_source_dependent_gaussian_jumping_distribution.py
Author: Keith Tauscher
Date: 6 Oct 2018

Description: Example of using the SourceDependentGaussianJumpingDistribution
             class to represent 1D Gaussian random variates.
"""
import os, time
import numpy as np
import matplotlib.pyplot as pl
from distpy import SourceDependentGaussianJumpingDistribution,\
    load_jumping_distribution_from_hdf5_file

sample_size = int(1e5)
umean1 = -1.
umean2 = 15.
uvar1 = 2.5
uvar2 = 10
distribution =\
    SourceDependentGaussianJumpingDistribution([-2, 10], [uvar1, uvar2])
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
    'univariate source-dependent Gaussian jumping distribution.').format(\
    time.time() - t0, sample_size))
pl.figure()
pl.hist(sample1, bins=100, histtype='step', color='b', linewidth=2,\
    label='sampled', density=True)
pl.hist(sample2, bins=100, histtype='step', color='b', linewidth=2,\
    label='sampled', density=True)
xlim = pl.xlim()
xs = np.linspace(xlim[0], xlim[1], 1000)
y1s = list(map((lambda x : np.exp(distribution.log_value(umean1, x))), xs))
y2s = list(map((lambda x : np.exp(distribution.log_value(umean2, x))), xs))
pl.plot(xs, y1s, linewidth=2, color='r', label='e^(log_prior)')
pl.plot(xs, y2s, linewidth=2, color='r', label='e^(log_prior)')
ylim = pl.ylim()
pl.ylim(ylim)
pl.title(('Univariate source-dependent Gaussian distributions variances=' +\
    '({0:.4g},{1:.4g})').format(uvar1, uvar2), size='xx-large')
pl.xlabel('Value', size='xx-large')
pl.ylabel('PDF', size='xx-large')
pl.tick_params(labelsize='xx-large', width=2, length=6)
pl.legend(fontsize='xx-large')
pl.show()

