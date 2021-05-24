"""
Example showcasing use of `SparseGaussianDistribution` for multidimensional
random variates with sparse covariance matrices.

**File**: $DISTPY/examples/distribution/sparse_gaussian_distribution.py  
**Author**: Keith Tauscher  
**Date**: May 14 2021
"""
import os, time
import numpy as np
import scipy.linalg as scila
import matplotlib.pyplot as pl
import matplotlib.cm as cm
from distpy import SparseSquareBlockDiagonalMatrix, SparseGaussianDistribution

def_cm = cm.bone
sample_size = int(1e5)

mean = [-7., 20.]
covariance = SparseSquareBlockDiagonalMatrix([[[125.]], [[125.]]])
distribution = SparseGaussianDistribution(mean, covariance)
hdf5_file_name = 'TEST_DELETE_THIS.hdf5'
distribution.save(hdf5_file_name)
try:
    assert distribution == SparseGaussianDistribution.load(hdf5_file_name)
except:
    os.remove(hdf5_file_name)
    raise
else:
    os.remove(hdf5_file_name)
assert(distribution.numparams == 2)

sparse_221 = SparseGaussianDistribution([1, 2, 3, 4, 5],\
    SparseSquareBlockDiagonalMatrix([np.identity(2), np.identity(2),\
    np.identity(1)]))
sparse_2 = SparseGaussianDistribution([1, 2],\
    SparseSquareBlockDiagonalMatrix([np.identity(2)]))
sparse_12 = SparseGaussianDistribution([3, 4, 5],\
    SparseSquareBlockDiagonalMatrix([np.identity(1), np.identity(2)]))
sparse_212 = SparseGaussianDistribution.combine(sparse_2, sparse_12)
assert(sparse_221 == sparse_212)

summand_distributions = []
summand_distributions.append(SparseGaussianDistribution([1, 2, 3, 4, 5],\
    SparseSquareBlockDiagonalMatrix([[[1]], [[2, 0, 0], [0, 3, 0], [0, 0, 4]],\
    [[5]]])))
summand_distributions.append(SparseGaussianDistribution([1, 2, 3, 4, 5],\
    SparseSquareBlockDiagonalMatrix([[[1, 0], [0, 2]], [[3, 0], [0, 4]],\
    [[5]]])))
sum_distribution = summand_distributions[0] + summand_distributions[1]
expected_sum_distribution = SparseGaussianDistribution(\
    [2, 4, 6, 8, 10], SparseSquareBlockDiagonalMatrix([[[2]], [[4]], [[6]],\
    [[8]], [[10]]]))
assert(sum_distribution == expected_sum_distribution)

t0 = time.time()
sample = distribution.draw(sample_size)
print(('It took {0:.5f} s for a sample of size {1} to be drawn from a ' +\
    'multivariate ({2}-parameter) gaussian').format(time.time() - t0,\
    sample_size, distribution.numparams))
print("sample_mean={0}, expected_mean={1}".format(np.mean(sample, axis=0),\
    distribution.mean))
print("sample_variance={0}, expected_variance={1}".format(\
    np.cov(sample, rowvar=False), distribution.variance))
mgp_xs = [sample[i][0] for i in range(sample_size)]
mgp_ys = [sample[i][1] for i in range(sample_size)]
pl.figure()
pl.hist2d(mgp_xs, mgp_ys, bins=100, cmap=def_cm)
pl.title(('SparseGaussian distribution (2 dimensions) with mean={0!s} and ' +\
    'covariance={1!s}').format(mean, covariance.dense()), size='xx-large')
pl.xlabel('x', size='xx-large')
pl.ylabel('y', size='xx-large')
pl.tick_params(labelsize='xx-large', width=2, length=6)
xs = np.arange(-50, 41)
ys = np.arange(-25, 66)
row_size = len(xs)
(xs, ys) = np.meshgrid(xs, ys)
logvalues = np.ndarray(xs.shape)
for ix in range(row_size):
    for iy in range(row_size):
        logvalues[ix,iy] = distribution.log_value([xs[ix,iy], ys[ix,iy]])
pl.figure()
pl.imshow(np.exp(logvalues), cmap=def_cm, extent=[-50.,40.,-25.,65.],\
    origin='lower')
pl.title('e^(log_value) for GaussianPrior', size='xx-large')
pl.xlabel('x', size='xx-large')
pl.ylabel('y', size='xx-large')
pl.tick_params(labelsize='xx-large', width=2, length=6)
pl.show()

