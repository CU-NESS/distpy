import time
import numpy as np
import matplotlib.pyplot as pl
from distpy import TruncatedGaussianDistribution

sample_size = int(1e5)

distribution = TruncatedGaussianDistribution(0., 1., -2., 1.)
assert distribution.numparams == 1
t0 = time.time()
sample = [distribution.draw() for i in xrange(sample_size)]
print ('It took %.5f s to draw %i ' % (time.time()-t0,sample_size,)) +\
       'points from a truncated Gaussian distribution.'
pl.figure()
pl.hist(sample, bins=100, linewidth=2, color='b', histtype='step',\
    label='sampled', normed=True)
xs = np.arange(-2.5, 2.501, 0.001)
pl.plot(xs, map((lambda x : np.exp(distribution.log_value(x))), xs),\
    linewidth=2, color='r', label='e^(log_value)')
pl.title('Truncated Gaussian distribution test', size='xx-large')
pl.xlabel('Value', size='xx-large')
pl.ylabel('PDF', size='xx-large')
pl.tick_params(labelsize='xx-large', width=2, length=6)
pl.legend(fontsize='xx-large')
pl.show()

