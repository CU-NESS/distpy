"""
File: examples/distribution/poisson_distribution.py
Author: Keith Tauscher
Date: 7 Aug 2017

Description: Example of using the PoissonDistribution class.
"""
import os, time
import numpy as np
import matplotlib.pyplot as pl
from distpy import PoissonDistribution

sample_size = int(1e5)

distribution = PoissonDistribution(10.)
assert distribution.numparams == 1
hdf5_file_name = 'TEST_DELETE_THIS.hdf5'
distribution.save(hdf5_file_name)
try:
    assert distribution == PoissonDistribution.load(hdf5_file_name)
except:
    os.remove(hdf5_file_name)
    raise
else:
    os.remove(hdf5_file_name)
t0 = time.time()
sample = distribution.draw(sample_size)
print(('It took {0:.5f} s to draw {1} points from a Poisson ' +\
    'distribution.').format(time.time() - t0, sample_size))
fig = pl.figure()
ax = fig.add_subplot(111)
ax.hist(sample, bins=np.arange(-0.5, 25.5, 1), histtype='step',\
    color='b', linewidth=2, density=True, label='sampled')
(start, end) = (0, 25)
xs = np.linspace(start, end, end - start + 1).astype(int)
distribution.plot(xs, ax=ax, show=False, linewidth=2, color='r',\
    label='e^(log_value)')
ax.legend(fontsize='xx-large', loc='upper right')
ax.set_title('Poisson distribution test', size='xx-large')
ax.set_xlabel('Value', size='xx-large')
ax.set_ylabel('PDF', size='xx-large')
ax.set_xlim((start, end))
ax.tick_params(labelsize='xx-large', width=2, length=6)
pl.show()

