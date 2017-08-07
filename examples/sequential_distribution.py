"""
File: examples/sequential_distribution.py
Author: Keith Tauscher
Date: 7 Aug 2017

Description: Example of using the SequentialDistribution class.
"""
import time
import numpy as np
import matplotlib.pyplot as pl
import matplotlib.cm as cm
from distpy import SequentialDistribution, UniformDistribution

def_cm = cm.bone
sample_size = int(1e5)

distribution = SequentialDistribution(UniformDistribution(0., 1.), 2)
t0 = time.time()
sample = [distribution.draw() for i in xrange(sample_size)]
print ("It took %.3f s to draw %i" % (time.time()-t0,sample_size,)) +\
      " vectors from a SequentialPrior with a Unif(0,1) distribution."
sam_xs = [sample[i][0] for i in xrange(sample_size)]
sam_ys = [sample[i][1] for i in xrange(sample_size)]
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
for ix in xrange(row_size):
    for iy in xrange(row_size):
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

