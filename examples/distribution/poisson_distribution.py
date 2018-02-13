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
pl.figure()
pl.hist(sample, bins=np.arange(-0.5, 25.5, 1), histtype='step',\
    color='b', linewidth=2, normed=True, label='sampled')
(start, end) = (0, 25)
xs = np.linspace(start, end, end - start + 1).astype(int)
pl.plot(xs, list(map((lambda x : np.exp(distribution.log_value(x))), xs)),\
    linewidth=2, color='r', label='e^(log_value)')
pl.legend(fontsize='xx-large', loc='upper right')
pl.title('Poisson distribution test', size='xx-large')
pl.xlabel('Value', size='xx-large')
pl.ylabel('PDF', size='xx-large')
pl.xlim((start, end))
pl.tick_params(labelsize='xx-large', width=2, length=6)
pl.show()

