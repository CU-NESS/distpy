"""
File: examples/distribution/multivariate_gaussian_distribution.py
Author: Keith Tauscher
Date: 10 Sep 2017

Description: Example showcasing use of DeterministicDistribution for
             multidimensional non-random variates.
"""
import os, time
import numpy as np
import matplotlib.pyplot as pl
import matplotlib.cm as cm
from distpy import DeterministicDistribution, load_distribution_from_hdf5_file

def_cm = cm.bone

repeats = 2000
nbins = 5
points = np.array([[0, 0], [-1, 1], [1, 1], [1, -1], [-1, -1]])
sample_size = points.shape[0] * repeats
points = np.repeat(points, repeats, axis=0)
distribution = DeterministicDistribution(points)
hdf5_file_name = 'TEST_DELETE_THIS.hdf5'
distribution.save(hdf5_file_name)
try:
    assert distribution == DeterministicDistribution.load(hdf5_file_name)
    assert distribution == load_distribution_from_hdf5_file(hdf5_file_name)
except:
    os.remove(hdf5_file_name)
    raise
else:
    os.remove(hdf5_file_name)
assert distribution.numparams == 2
t0 = time.time()
sample = distribution.draw(sample_size)
print(('It took {0:.5f} s for a sample of size {1} to be drawn from a ' +\
    'multivariate ({2}-parameter) deterministic distribution').format(\
    time.time() - t0, sample_size, distribution.numparams))
(sample_xs, sample_ys) = (sample[:,0], sample[:,1])
pl.figure()
pl.hist2d(sample_xs, sample_ys, bins=nbins, cmap=def_cm)
pl.title('Multivariate deterministic distribution (2 dimensions)',\
    size='xx-large')
pl.xlabel('x', size='xx-large')
pl.ylabel('y', size='xx-large')
pl.tick_params(labelsize='xx-large', width=2, length=6)
pl.show()

