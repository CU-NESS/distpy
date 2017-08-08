"""
File: examples/parallelepiped_distribution.py
Author: Keith Tauscher
Date: 7 Aug 2017

Description: Example of using the ParallelepipedDistribution class.
"""
import time
import numpy as np
import matplotlib.pyplot as pl
import matplotlib.cm as cm
from distpy import ParallelepipedDistribution

def_cm = cm.bone
sample_size = int(1e5)

center = [-15., 20.]
face_directions = [[1., 1.], [1., -1.]]
distances = [10., 1.]
distribution = ParallelepipedDistribution(center, face_directions, distances,\
    norm_dirs=False)
assert distribution.numparams == 2
t0 = time.time()
sample = distribution.draw(sample_size)
print (('It took %.5f s to draw %i ' % (time.time()-t0,sample_size,)) +\
      'points from a bivariate parallelogram shaped uniform distribution.')
xs = [sample[i][0] for i in range(sample_size)]
ys = [sample[i][1] for i in range(sample_size)]
pl.figure()
pl.hist2d(xs, ys, bins=100, cmap=def_cm)
pl.title('Parallelogram shaped uniform dist. centered at %s.' % center,\
    size='xx-large')
pl.xlabel('x', size='xx-large')
pl.ylabel('y', size='xx-large')
pl.tick_params(labelsize='xx-large', width=2, length=6)
xs = np.arange(-20.5, -9.4, 0.1)
ys = np.arange(14.5, 25.6, 0.1)
(xs, ys) = np.meshgrid(xs, ys)
(x_size, y_size) = xs.shape
logvalues = np.ndarray(xs.shape)
for ix in xrange(x_size):
    for iy in xrange(y_size):
        logvalues[ix,iy] = distribution.log_value([xs[ix,iy], ys[ix,iy]])
pl.figure()
pl.imshow(np.exp(logvalues), cmap=def_cm, extent=[-25.,-4.9,14.,26.1],\
    origin='lower')
pl.title('e^(log_value) for ParallelepipedPrior distribution',\
    size='xx-large')
pl.xlabel('x', size='xx-large')
pl.ylabel('y', size='xx-large')
pl.tick_params(labelsize='xx-large', width=2, length=6)
for point in [(-20.2, 16.3), (-20.1, 14.5), (-20., 14.6), (-19., 14.6),\
              (-11., 25.4), (-10., 25.4), (-9.6, 25.), (-9.6, 24.)]:
    assert distribution.log_value(point) == -np.inf
for point in [(-19.5, 14.6), (-20.4, 15.5), (-9.6, 24.5), (-10.5, 25.4)]:
    assert distribution.log_value(point) == distribution.log_value(center)
pl.show()

