"""
File: examples/distribution/binomial_distribution.py
Author: Keith Tauscher
Date: 7 Aug 2017

Description: Example of using the BinomialDistribution class.
"""
import time
import numpy as np
import matplotlib.pyplot as pl
from distpy import BinomialDistribution

sample_size = int(1e5)

distribution = BinomialDistribution(0.4, 10)
assert distribution.numparams == 1
t0 = time.time()
sample = distribution.draw(sample_size)
print(('It took {0:.5f} s to draw {1} points from a binomial ' +\
    'distribution.').format(time.time() - t0, sample_size))
pl.figure()
pl.hist(sample, bins=np.arange(-0.5, 11, 1), histtype='step', color='b',\
    linewidth=2, normed=True, label='sampled')
(start, end) = (0, 10)
xs = np.linspace(start, end, end - start + 1).astype(int)
pl.plot(xs, list(map((lambda x : np.exp(distribution.log_value(x))), xs)),\
    linewidth=2, color='r', label='e^(log_value)')
pl.legend(fontsize='xx-large', loc='upper right')
pl.title('Binomial distribution test', size='xx-large')
pl.xlabel('Value', size='xx-large')
pl.ylabel('PDF', size='xx-large')
pl.xlim((0, 10))
pl.tick_params(labelsize='xx-large', width=2, length=6)
pl.show()

