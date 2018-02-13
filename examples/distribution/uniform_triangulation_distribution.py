"""
File: examples/distribution/uniform_triangulation_distribution.py
Author: Keith Tauscher
Date: 7 Aug 2017

Description: Example of using the UniformTriangulationDistribution class.
"""
import os, time
import numpy as np
import matplotlib.pyplot as pl
import matplotlib.cm as cm
from distpy import UniformTriangulationDistribution

def_cm = cm.bone
sample_size = int(1e5)

points = np.array([[0, 0], [-1, 1], [1, 1]])
distribution = UniformTriangulationDistribution(points=points)
hdf5_file_name = 'TEST_DELETE_THIS.hdf5'
distribution.save(hdf5_file_name)
try:
    assert\
        distribution == UniformTriangulationDistribution.load(hdf5_file_name)
except:
    os.remove(hdf5_file_name)
    raise
else:
    os.remove(hdf5_file_name)
assert distribution.numparams == 2
t0 = time.time()
sample = distribution.draw(sample_size)
print(('It took {0:.5f} s to draw {1} points from a bivariate ' +\
    'triangle shaped unform distribution.').format(time.time() - t0,\
    sample_size))
xs = [sample[i][0] for i in range(sample_size)]
ys = [sample[i][1] for i in range(sample_size)]
pl.figure()
pl.hist2d(xs, ys, bins=100, cmap=def_cm)
pl.title('Triangle shaped uniform distribution with vertices {}.'.format(\
    points), size='xx-large')
pl.xlabel('x', size='xx-large')
pl.ylabel('y', size='xx-large')
pl.tick_params(labelsize='xx-large', width=2, length=6)
min_x = np.min(points[:,0])
max_x = np.max(points[:,0])
min_y = np.min(points[:,1])
max_y = np.max(points[:,1])
xs = np.linspace(min_x, max_x, 1000)
ys = np.linspace(min_y, max_y, 1000)
(xs, ys) = np.meshgrid(xs, ys)
(x_size, y_size) = xs.shape
logvalues = np.ndarray(xs.shape)
for ix in range(x_size):
    for iy in range(y_size):
        logvalues[ix,iy] = distribution.log_value([xs[ix,iy], ys[ix,iy]])
pl.figure()
pl.imshow(np.exp(logvalues), cmap=def_cm, extent=[min_x,max_x,min_y,max_y],\
    origin='lower')
pl.title('e^(log_value) for UniformTriangulationPrior distribution',\
    size='xx-large')
pl.xlabel('x', size='xx-large')
pl.ylabel('y', size='xx-large')
pl.tick_params(labelsize='xx-large', width=2, length=6)
pl.show()

