"""
File: examples/jumping/multivariate_gaussian_jumping_distribution.py
Author: Keith Tauscher
Date: 22 Dec 2017

Description: Example of using the GaussianJumpingDistribution class to
             represent 2D Gaussian random variates.
"""
import os, time
import numpy as np
import matplotlib.pyplot as pl
from distpy import UniformJumpingDistribution,\
    load_jumping_distribution_from_hdf5_file

cmap = 'bone'
sample_size = int(1e5)
umean1 = np.array([0, 0])
umean2 = np.array([1, 1])
uvar = np.array([[4., 0.], [0., 1.]])
distribution = UniformJumpingDistribution(uvar)
assert distribution.numparams == 2
try:
    file_name = 'TEMPORARY_TEST_DELETE_THIS_IF_IT_EXISTS.hdf5'
    distribution.save(file_name)
    loaded_distribution = load_jumping_distribution_from_hdf5_file(file_name)
    assert loaded_distribution == distribution
except:
    os.remove(file_name)
    raise
else:
    os.remove(file_name)
t0 = time.time()
sample1 = distribution.draw(umean1, sample_size)
sample2 = distribution.draw(umean2, sample_size)
print(('It took {0:.5f} s for two samples of size {1} to be drawn from a ' +\
    '2D uniform elliptical jumping distribution.').format(time.time() - t0,\
    sample_size))
pl.figure()
pl.hist2d(sample1[:,0], sample1[:,1], cmap=cmap, bins=100)
pl.title('Centered on {} (sampled)'.format(umean1), size='xx-large')
xlim1 = pl.xlim()
ylim1 = pl.ylim()
x1s = np.linspace(xlim1[0], xlim1[1], 100)
y1s = np.linspace(ylim1[0], ylim1[1], 100)
extent1 = [np.min(x1s), np.max(x1s), np.min(y1s), np.max(y1s)]
(x1s, y1s) = np.meshgrid(x1s, y1s)
z1s = np.exp(np.reshape([distribution.log_value(umean1, np.array([x, y])) for (x, y) in zip(x1s.flatten(), y1s.flatten())], (100, 100)))
pl.figure()
pl.imshow(z1s, cmap=cmap, extent=extent1)
pl.title('Centered on {} (expected)'.format(umean1), size='xx-large')


pl.figure()
pl.hist2d(sample2[:,0], sample2[:,1], cmap=cmap, bins=100)
pl.title('Centered on {} (sampled)'.format(umean2), size='xx-large')
xlim2 = pl.xlim()
ylim2 = pl.ylim()
x2s = np.linspace(xlim2[0], xlim2[1], 100)
y2s = np.linspace(ylim2[0], ylim2[1], 100)
extent2 = [np.min(x2s), np.max(x2s), np.min(y2s), np.max(y2s)]
(x2s, y2s) = np.meshgrid(x2s, y2s)
z2s = np.exp(np.reshape([distribution.log_value(umean2, np.array([x, y])) for (x, y) in zip(x2s.flatten(), y2s.flatten())], (100, 100)))
pl.figure()
pl.imshow(z2s, cmap=cmap, extent=extent2)
pl.title('Centered on {} (expected)'.format(umean2), size='xx-large')
pl.tick_params(labelsize='xx-large', width=2, length=6)
pl.show()


