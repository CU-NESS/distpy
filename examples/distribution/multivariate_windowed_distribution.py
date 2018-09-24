"""
File: examples/distribution/multivariate_windowed_distribution.py
Author: Keith Tauscher
Date: 23 Sep 2018

Description: Example of drawing uniformly from the intersection of two
             perpendicularly oriented diagonal ellipses in 2D space.
"""
import os, time
import numpy as np
import matplotlib.pyplot as pl
import matplotlib.cm as cm
from distpy import EllipticalUniformDistribution, WindowedDistribution,\
    load_distribution_from_hdf5_file

def_cm = cm.bone
sample_size = int(1e5)

ellmean = [0, 0]
ellcov1 = [[2.5, 1.5], [1.5, 2.5]]
ellcov2 = [[2.5, -1.5], [-1.5, 2.5]]
elliptical_distribution1 = EllipticalUniformDistribution(ellmean, ellcov1)
elliptical_distribution2 = EllipticalUniformDistribution(ellmean, ellcov2)
distribution =\
    WindowedDistribution(elliptical_distribution1, elliptical_distribution2)
hdf5_file_name = 'TEST_DELETE_THIS.hdf5'
distribution.save(hdf5_file_name)
try:
    assert(distribution == WindowedDistribution.load(hdf5_file_name,\
        EllipticalUniformDistribution, EllipticalUniformDistribution))
    assert(distribution == load_distribution_from_hdf5_file(hdf5_file_name))
except:
    os.remove(hdf5_file_name)
    raise
else:
    os.remove(hdf5_file_name)
t0 = time.time()
sample = distribution.draw(sample_size)
print(("It took {0:.5f} s for a sample of size {1} to be drawn from a " +\
    "uniform prior over the intersection of two ellipses.").format(\
    time.time() - t0, sample_size))
ellp_xs = [sample[idraw][0] for idraw in range(sample_size)]
ellp_ys = [sample[idraw][1] for idraw in range(sample_size)]
fig = pl.figure(figsize=(12,9))
ax = fig.add_subplot(111)
ax.hist2d(ellp_xs, ellp_ys, bins=50, cmap=def_cm)
ax.set_title('Multivariate elliptical intersection distribution (2 ' +\
    'dimensions)', size='xx-large')
ax.set_xlabel('x', size='xx-large')
ax.set_ylabel('y', size='xx-large')
ax.tick_params(labelsize='xx-large', width=2, length=6)

pl.show()

