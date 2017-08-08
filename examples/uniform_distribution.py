"""
File: examples/uniform_distribution.py
Author: Keith Tauscher
Date: 7 Aug 2017

Description: Example of using the UniformDistribution.
"""
import time
import numpy as np
import matplotlib.pyplot as pl
from distpy import UniformDistribution

low = -27.
high = 19.
sample_size = int(1e5)

distribution = UniformDistribution(high, low)
distribution2 = UniformDistribution(low, high)
assert distribution.numparams == 1
assert (distribution.low == distribution2.low) and\
    (distribution.high == distribution2.high)
t0 = time.time()
sample = distribution.draw(sample_size)
print ('It took %.5f s to draw %i' % (time.time()-t0, sample_size)) +\
      ' points from a univariate uniform distribution.'
pl.figure()
pl.hist(sample, bins=100, histtype='step', color='b', linewidth=2,\
    normed=True, label='sampled')
xs = np.arange(-30., 20., 0.01)
pl.plot(xs, map((lambda x : np.exp(distribution.log_value(x))), xs),\
    linewidth=2, color='r', label='e^(log_value)')
pl.title('Uniform distribution on ' +\
         '[%s,%s]' % (distribution.low, distribution.high,), size='xx-large')
pl.xlabel('Value', size='xx-large')
pl.ylabel('PDF', size='xx-large')
pl.tick_params(labelsize='xx-large', width=2, length=6)
pl.legend(fontsize='xx-large', loc='lower center')
pl.show()

