"""
File: examples/distribution/binomial_distribution.py
Author: Keith Tauscher
Date: Oct 15 2019

Description: Example of using the BinomialDistribution class.
"""
import os, time
import numpy as np
import matplotlib.pyplot as pl
from distpy import BinomialDistribution

sample_size = int(1e5)

distribution = BinomialDistribution(0.4, 10)
hdf5_file_name = 'TEST_DELETE_THIS.hdf5'
distribution.save(hdf5_file_name)
try:
    assert distribution == BinomialDistribution.load(hdf5_file_name)
except:
    os.remove(hdf5_file_name)
    raise
else:
    os.remove(hdf5_file_name)
assert distribution.numparams == 1
t0 = time.time()
sample = distribution.draw(sample_size)
print(('It took {0:.5f} s to draw {1} points from a binomial ' +\
    'distribution.').format(time.time() - t0, sample_size))
print('Sample mean was {0:.3g}, while expected mean was {1:.3g}.'.format(\
    np.mean(sample), distribution.mean))
print(('Sample standard deviation was {0:.3g}, while expected standard ' +\
    'deviation was {1:.3g}.').format(np.std(sample),\
    distribution.standard_deviation))
fig = pl.figure()
ax = fig.add_subplot(111)
ax.hist(sample, bins=np.arange(-0.5, 11, 1), histtype='step', color='b',\
    linewidth=2, density=True, label='sampled')
(start, end) = (0, 10)
xs = np.linspace(start, end, end - start + 1).astype(int)
distribution.plot(xs, ax=ax, show=False, linewidth=2, color='r',\
    label='e^(log_value)')
ax.legend(fontsize='xx-large', loc='upper right')
ax.set_title('Binomial distribution test', size='xx-large')
ax.set_xlabel('Value', size='xx-large')
ax.set_ylabel('PDF', size='xx-large')
ax.set_xlim((-0.5, 10.5))
ax.tick_params(labelsize='xx-large', width=2, length=6)
pl.show()

