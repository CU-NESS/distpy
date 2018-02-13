"""
File: examples/distribution/geometric_distribution.py
Author: Keith Tauscher
Date: 8 January 2018

Description: Example of using the GeometricDistribution class including the
             minimum and maximum.
"""
import os, time
import numpy as np
import matplotlib.pyplot as pl
from distpy import GeometricDistribution

sample_size = int(1e5)

common_ratio = 0.95
minimum = 3
maximum = 20
metadata = {'a': "This is pretty complicated", 'b': 1,\
    'c': True, 'd': "metadata don't you think?"}
distribution = GeometricDistribution(common_ratio, minimum=minimum,\
    maximum=maximum, metadata=metadata)
assert distribution.numparams == 1
hdf5_file_name = 'TEST_DELETE_THIS.hdf5'
distribution.save(hdf5_file_name)
try:
    assert distribution == GeometricDistribution.load(hdf5_file_name)
except:
    #	os.remove(hdf5_file_name)
    raise
else:
    os.remove(hdf5_file_name)
t0 = time.time()
sample = distribution.draw(sample_size)
print(('It took {0:.5f} s to draw {1} points from a geometric ' +\
    'distribution.').format(time.time() - t0, sample_size))
pl.figure()
(start, end) = (minimum - 3, maximum + 3)
bins = np.arange(start, end + 1)
bins = (bins[1:] + bins[:-1]) / 2.
pl.hist(sample, bins=np.arange(start - 0.5, end + 3.5, 1), histtype='step',\
    color='b', linewidth=2, normed=True, label='sampled')
xs = np.linspace(start, end, end - start + 1).astype(int)
pl.scatter(xs, list(map((lambda x : np.exp(distribution.log_value(x))), xs)),\
    linewidth=2, color='r', label='e^(log_value)')
pl.legend(fontsize='xx-large', loc='upper right')
pl.title('Geometric distribution test', size='xx-large')
pl.xlabel('Value', size='xx-large')
pl.ylabel('PDF', size='xx-large')
pl.xlim((start, end))
pl.tick_params(labelsize='xx-large', width=2, length=6)
pl.show()

