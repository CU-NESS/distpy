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
    os.remove(hdf5_file_name)
    raise
else:
    os.remove(hdf5_file_name)
t0 = time.time()
sample = distribution.draw(sample_size)
print(('It took {0:.5f} s to draw {1} points from a geometric ' +\
    'distribution.').format(time.time() - t0, sample_size))
fig = pl.figure()
ax = fig.add_subplot(111)
(start, end) = (minimum - 3, maximum + 3)
bins = np.arange(start, end + 1)
bins = (bins[1:] + bins[:-1]) / 2.
ax.hist(sample, bins=np.arange(start - 0.5, end + 3.5, 1), histtype='step',\
    color='b', linewidth=2, density=True, label='sampled')
xs = np.linspace(start, end, end - start + 1).astype(int)
distribution.plot(xs, ax=ax, show=False, linewidth=2, color='r',\
    label='e^(log_value)')
ax.legend(fontsize='xx-large', loc='upper right')
ax.set_title('Geometric distribution test', size='xx-large')
ax.set_xlabel('Value', size='xx-large')
ax.set_ylabel('PDF', size='xx-large')
ax.set_xlim((start, end))
ax.tick_params(labelsize='xx-large', width=2, length=6)
pl.show()

