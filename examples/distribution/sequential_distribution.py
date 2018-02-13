"""
File: examples/distribution/sequential_distribution.py
Author: Keith Tauscher
Date: 7 Aug 2017

Description: Example of using the SequentialDistribution class.
"""
import os, time
import numpy as np
import matplotlib.pyplot as pl
import matplotlib.cm as cm
from distpy import SequentialDistribution, UniformDistribution

def_cm = cm.bone
sample_size = int(1e5)

distribution = SequentialDistribution(UniformDistribution(0., 1.), 2)
hdf5_file_name = 'TEST_DELETE_THIS.hdf5'
distribution.save(hdf5_file_name)
try:
    assert distribution ==\
        SequentialDistribution.load(hdf5_file_name, UniformDistribution)
except:
    os.remove(hdf5_file_name)
    raise
else:
    os.remove(hdf5_file_name)
t0 = time.time()
sample = distribution.draw(sample_size)
print(("It took {0:.5f} s to draw {1} vectors from a " +\
    "SequentialDistribution with a Unif(0,1) distribution.").format(\
    time.time() - t0, sample_size))
sam_xs = [sample[i][0] for i in range(sample_size)]
sam_ys = [sample[i][1] for i in range(sample_size)]
pl.figure()
pl.hist2d(sam_xs, sam_ys, bins=100, cmap=def_cm)
pl.title('Sampled distribution of a LinkedPrior ' +\
         'with a Unif(0,1) distribution', size='xx-large')
pl.xlabel('x', size='xx-large')
pl.ylabel('y', size='xx-large')
pl.tick_params(labelsize='xx-large', width=2, length=6)
xs = np.arange(0., 1.01, 0.01)
ys = np.arange(0., 1.01, 0.01)
row_size = len(xs)
(xs, ys) = np.meshgrid(xs, ys)
logvalues = np.ndarray(xs.shape)
for ix in range(row_size):
    for iy in range(row_size):
        logvalues[ix,iy] = distribution.log_value([xs[ix,iy], ys[ix,iy]])
pl.figure()
pl.imshow(np.exp(logvalues), cmap=def_cm, extent=[0.,1.,0.,1.],\
    origin='lower')
pl.title('e^(log_value) for SequentialPrior with Unif(0,1) distribution',\
    size='xx-large')
pl.xlabel('x', size='xx-large')
pl.ylabel('y', size='xx-large')
pl.tick_params(labelsize='xx-large', width=2, length=6)
pl.show()

