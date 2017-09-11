"""
File: examples/distribution/elliptical_uniform_distribution.py
Author: Keith Tauscher
Date: 7 Aug 2017

Description: Example of using the EllipticalUniformDistribution class.
"""
import time
import numpy as np
import matplotlib.pyplot as pl
import matplotlib.cm as cm
from distpy import EllipticalUniformDistribution

def_cm = cm.bone
sample_size = int(1e5)

ellmean = [4.76, -12.64]
ellcov = [[1, -0.5], [-0.5, 1]]
distribution = EllipticalUniformDistribution(ellmean, ellcov)
t0 = time.time()
sample = distribution.draw(sample_size)
print (("It took {0:.5f} s for a sample of size {1} to be drawn from a " +\
    "uniform multivariate elliptical prior.").format(time.time() - t0,\
    sample_size))
ellp_xs = [sample[idraw][0] for idraw in range(sample_size)]
ellp_ys = [sample[idraw][1] for idraw in range(sample_size)]
pl.figure()
pl.hist2d(ellp_xs, ellp_ys, bins=50, cmap=def_cm)
pl.title('Multivariate elliptical distribution (2 dimensions) with ' +\
    ('mean=%s and covariance=%s' % (ellmean, ellcov,)), size='xx-large')
pl.xlabel('x', size='xx-large')
pl.ylabel('y', size='xx-large')
pl.tick_params(labelsize='xx-large', width=2, length=6)
xs = np.arange(2.7, 6.9, 0.05)
ys = np.arange(-14.7, -10.5, 0.05)
row_size = len(xs)
(xs, ys) = np.meshgrid(xs, ys)
logvalues = np.ndarray(xs.shape)
for ix in range(row_size):
    for iy in range(row_size):
        logvalues[ix,iy] = distribution.log_value([xs[ix,iy], ys[ix,iy]])
pl.figure()
pl.imshow(np.exp(logvalues), cmap=def_cm, extent=[2.7, 6.85,-14.7,-10.45],\
    origin='lower')
pl.title('e^(log_value) for EllipticalPrior',\
    size='xx-large')
pl.xlabel('x', size='xx-large')
pl.ylabel('y', size='xx-large')
pl.tick_params(labelsize='xx-large', width=2, length=6)
pl.show()
