import time
import numpy as np
import matplotlib.pyplot as pl
import matplotlib.cm as cm
from distpy import GaussianDistribution

def_cm = cm.bone
sample_size = int(1e5)

mmean = [-7., 20.]
mcov = [[125., 75.], [75., 125.]]
distribution = GaussianDistribution(mmean, mcov)
assert distribution.numparams == 2
t0 = time.time()
sample = [distribution.draw() for i in xrange(sample_size)]
print (('It took %.3f s for a sample ' % (time.time()-t0)) +\
      ('of size %i to be drawn from a multivariate ' % (sample_size,)) +\
      ('(%i parameters) Gaussian.' % distribution.numparams))
mgp_xs = [sample[i][0] for i in range(sample_size)]
mgp_ys = [sample[i][1] for i in range(sample_size)]
pl.figure()
pl.hist2d(mgp_xs, mgp_ys, bins=100, cmap=def_cm)
pl.title('Multivariate Gaussian prior (2 dimensions) with ' +\
         ('mean=%s and covariance=%s' % (mmean, mcov)), size='xx-large')
pl.xlabel('x', size='xx-large')
pl.ylabel('y', size='xx-large')
pl.tick_params(labelsize='xx-large', width=2, length=6)
xs = np.arange(-50., 40.1, 0.1)
ys = np.arange(-25., 65.1, 0.1)
row_size = len(xs)
(xs, ys) = np.meshgrid(xs, ys)
logvalues = np.ndarray(xs.shape)
for ix in xrange(row_size):
    for iy in xrange(row_size):
        logvalues[ix,iy] = distribution.log_value([xs[ix,iy], ys[ix,iy]])
pl.figure()
pl.imshow(np.exp(logvalues), cmap=def_cm, extent=[-50.,40.,-25.,65.],\
    origin='lower')
pl.title('e^(log_value) for GaussianPrior', size='xx-large')
pl.xlabel('x', size='xx-large')
pl.ylabel('y', size='xx-large')
pl.tick_params(labelsize='xx-large', width=2, length=6)
pl.show()

