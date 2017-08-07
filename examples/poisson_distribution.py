"""
File: examples/poisson_distribution.py
Author: Keith Tauscher
Date: 7 Aug 2017

Description: Example of using the PoissonDistribution class.
"""
import time
import numpy as np
import matplotlib.pyplot as pl
from distpy import PoissonDistribution

sample_size = int(1e5)

distribution = PoissonDistribution(10.)
assert distribution.numparams == 1
t0 = time.time()
sample = [distribution.draw() for i in xrange(sample_size)]
print ('It took %.5f s to draw %i ' % (time.time() - t0, sample_size)) +\
      'points from a Poisson distribution.'
pl.figure()
pl.hist(sample, bins=np.arange(-0.5, 25.5, 1), histtype='step',\
    color='b', linewidth=2, normed=True, label='sampled')
(start, end) = (0, 25)
xs = np.linspace(start, end, end - start + 1).astype(int)
pl.plot(xs, map((lambda x : np.exp(distribution.log_value(x))), xs),\
    linewidth=2, color='r', label='e^(log_value)')
pl.legend(fontsize='xx-large', loc='upper right')
pl.title('Poisson distribution test', size='xx-large')
pl.xlabel('Value', size='xx-large')
pl.ylabel('PDF', size='xx-large')
pl.xlim((start, end))
pl.tick_params(labelsize='xx-large', width=2, length=6)
pl.show()

