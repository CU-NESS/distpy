"""
File: examples/chi_squared_distribution.py
Author: Keith Tauscher
Date: 7 Aug 2017

Description: Example of using the ChiSquaredDistribution class.
"""
import time
import numpy as np
import matplotlib.pyplot as pl
from distpy import ChiSquaredDistribution

sample_size = int(1e5)

distribution = ChiSquaredDistribution(5)
assert distribution.numparams == 1
t0 = time.time()
sample = distribution.draw(sample_size)
print 'It took %.5f s to draw %i points from a chisquared distribution.' %\
    (time.time()-t0,sample_size,)
pl.figure()
pl.hist(sample, bins=100, linewidth=2, color='b', histtype='step',\
    label='sampled', normed=True)
xs = np.arange(0.01, 30., 0.01)
pl.plot(xs, map((lambda x : np.exp(distribution.log_value(x))), xs),\
    linewidth=2, color='r', label='e^(log_value)')
pl.title('Chi-squared distribution test', size='xx-large')
pl.xlabel('Value', size='xx-large')
pl.ylabel('PDF', size='xx-large')
pl.tick_params(labelsize='xx-large', width=2, length=6)
pl.legend(fontsize='xx-large')
pl.show()

