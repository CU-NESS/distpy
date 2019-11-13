"""
File: examples/distribution/multivariate_custom_discrete_distribution.py
Author: Keith Tauscher
Date: Oct 15 2019

Description: Example showing how to use the CustomDiscreteDistribution class to
             represent a multivative discrete distribution with a simple
             (linear) varying probability mass function.
"""
import os, time
import numpy as np
import matplotlib.pyplot as pl
from distpy import CustomDiscreteDistribution, load_distribution_from_hdf5_file

cmap = 'bone'

fontsize = 24
nxvalues = 15
nyvalues = 10
x_values = np.arange(nxvalues) + 1
y_values = np.arange(nyvalues) + 1
xbins = np.concatenate([x_values - 0.5, [x_values[-1] + 0.5]])
ybins = np.concatenate([y_values - 0.5, [y_values[-1] + 0.5]])
xlim = (xbins[0], xbins[-1])
ylim = (ybins[0], ybins[-1])
pmf_values = (x_values[:,np.newaxis] + y_values[np.newaxis,:] + 10)
distribution = CustomDiscreteDistribution((x_values, y_values), pmf_values)
hdf5_file_name = 'TEST_DELETE_THIS.hdf5'
distribution.save(hdf5_file_name)
try:
    assert distribution == load_distribution_from_hdf5_file(hdf5_file_name)
except:
    os.remove(hdf5_file_name)
    raise
else:
    os.remove(hdf5_file_name)

log_values = np.ndarray((len(x_values), len(y_values)))
for ix in range(nxvalues):
    for iy in range(nyvalues):
        log_values[ix,iy] =\
            distribution.log_value((x_values[ix], y_values[iy]))

ndraw = int(1e6)
t0 = time.time()
draws = distribution.draw(ndraw)
t1 = time.time()
print(("It took {0:.5f} s to draw {1:d} samples from a 2D custom discrete " +\
    "distribution with {2:d} pixels.").format(t1 - t0, ndraw,\
    nxvalues * nyvalues))
print("sample_mean={0}, expected_mean={1}".format(np.mean(draws, axis=0),\
    distribution.mean))
print("sample_variance={0}, expected_variance={1}".format(\
    np.cov(draws, rowvar=False), distribution.variance))

pl.figure()
pl.hist2d(draws[:,0], draws[:,1], bins=(xbins, ybins), normed=True, cmap=cmap)
pl.colorbar()
pl.xlim(xlim)
pl.ylim(ylim)
pl.xlabel('x', size=fontsize)
pl.ylabel('y', size=fontsize)
pl.tick_params(width=2.5, length=7.5, labelsize=fontsize)
pl.title('Observed', size=fontsize)

pl.figure()
pl.imshow(np.exp(log_values)[:,-1::-1].T, cmap=cmap, extent=list(xlim+ylim))
pl.colorbar()
pl.xlim(xlim)
pl.ylim(ylim)
pl.xlabel('x', size=fontsize)
pl.ylabel('y', size=fontsize)
pl.tick_params(width=2.5, length=7.5, labelsize=fontsize)
pl.title('Expected', size=fontsize)

pl.show()

