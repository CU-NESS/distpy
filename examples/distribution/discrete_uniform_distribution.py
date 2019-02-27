"""
File: examples/distribution/discrete_uniform_distribution.py
Author: Keith Tauscher
Date: 16 Jun 2018

Description: Example of using the DiscreteUniformDistribution.
"""
import os, time
import numpy as np
import matplotlib.pyplot as pl
from distpy import DiscreteUniformDistribution,\
    load_distribution_from_hdf5_file

low = -27
high = 19
sample_size = int(1e6)

distribution = DiscreteUniformDistribution(high, low, metadata=1)
distribution2 = DiscreteUniformDistribution(low, high)
hdf5_file_name = 'TEST_DELETE_THIS.hdf5'
distribution.save(hdf5_file_name)
try:
    assert (distribution == DiscreteUniformDistribution.load(hdf5_file_name))
    assert (distribution == load_distribution_from_hdf5_file(hdf5_file_name))
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
print(('It took {0:.5f} s to draw {1} points from a discrete uniform ' +\
    'distribution.').format(time.time() - t0, sample_size))
fig = pl.figure()
ax = fig.add_subplot(111)
bins = np.arange(low - 2, high + 3) - 0.5
ax.hist(sample, bins=bins, histtype='step', color='b', linewidth=2,\
    density=True, label='sampled')
xs = bins + 0.5
distribution.plot(xs, ax=ax, show=False, color='r', label='e^(log_value)')
ax.set_title(('Uniform distribution on [{0!s},{1!s}]').format(\
    distribution.low, distribution.high), size='xx-large')
ax.set_xlabel('Value', size='xx-large')
ax.set_ylabel('PDF', size='xx-large')
ax.tick_params(labelsize='xx-large', width=2, length=6)
ax.legend(fontsize='xx-large', loc='lower center')
pl.show()

