"""
File: examples/distribution/distribution_harmonizer.py
Author: Keith Tauscher
Date: 26 Feb 2019

Description: Example script showing how to use a DistributionHarmonizer object,
             which creates a consistent N-D distribution out of a M-D
             distribution and a solver which computes the remaining (N-M)
             parameters.
"""
import time
import numpy as np
import matplotlib.pyplot as pl
from distpy import GaussianDistribution, DistributionSet,\
    DistributionHarmonizer

fontsize = 24

known_distribution_set =\
    DistributionSet([(GaussianDistribution(0, 1), 'x', None)])
def solver(dictionary):
    return {'y': 1 + dictionary['x']}
ndraw = 10000

distribution_harmonizer =\
    DistributionHarmonizer(known_distribution_set, solver, ndraw)
start_time = time.time()
full_distribution_set = distribution_harmonizer.full_distribution_set
middle_time = time.time()
sample = full_distribution_set.draw(ndraw)
end_time = time.time()
form_duration = middle_time - start_time
draw_duration = end_time - middle_time
print(("It took {0:.5f} s to form the full distribution set with " +\
    "ndraw={1:d} from a DistributionHarmonizer and {2:.5f} s to return the " +\
    "sample from that distribution.").format(form_duration, ndraw,\
    draw_duration))
sample = np.array([sample['x'], sample['y']])

fig = pl.figure(figsize=(12, 9))
ax = fig.add_subplot(111)
ax.hist2d(sample[0], sample[1],\
    bins=(np.linspace(-3, 3, 10), np.linspace(-2, 4, 10)))
ax.set_xlabel('x', size=fontsize)
ax.set_ylabel('y', size=fontsize)
ax.tick_params(labelsize=fontsize, width=2.5, length=7.5, which='major')
ax.tick_params(labelsize=fontsize, width=1.5, length=4.5, which='minor')

pl.show()

