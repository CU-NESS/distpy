"""
File: examples/distribution/high_dimensional_gaussian.py
Author: Keith Tauscher
Date: 10 May 2018

Description: Example which calculates the time it takes to draw many separate
             times from the same multivariate Gaussian distribution and
             compares it to the dimension of the space.
"""
from __future__ import division
import time
import numpy as np
import matplotlib.pyplot as pl
from distpy import GaussianDistribution

fontsize = 20
ndraw = int(50)
ndims = np.arange(0, 100, 2) + 1

durations = []

for (indim, ndim) in enumerate(ndims):
    random_matrix = np.random.normal(size=(ndim, ndim))
    covariance = np.dot(random_matrix, random_matrix.T)
    distribution = GaussianDistribution(np.zeros(ndim), covariance)
    first_time = time.time()
    for idraw in range(ndraw):
        distribution.draw()
    second_time = time.time()
    durations.append(second_time - first_time)

log_ndims = np.log10(ndims)
log_durations = np.log10(durations)
pl.scatter(log_ndims, log_durations, label='data', color='k')
halfway_index = len(ndims) // 2
coeff = np.polyfit(log_ndims[halfway_index:], log_durations[halfway_index:], 1)
best_fit = np.polyval(coeff, log_ndims)
pl.plot(log_ndims, best_fit, color='k',\
    label='log10(t)={0:.4g}+{1:.4g}*log10(ndim)'.format(coeff[1], coeff[0]))
pl.xlabel('log10(t)', size=fontsize)
pl.ylabel('log10(ndim)', size=fontsize)
pl.title('log(t) vs log(ndim)', size=fontsize)
pl.tick_params(labelsize=fontsize, length=7.5, width=2.5, which='major')
pl.tick_params(length=4.5, width=1.5, which='minor')
pl.legend(fontsize=fontsize)
pl.show()

