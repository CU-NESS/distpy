"""
File: examples/distribution/univariate_windowed_distribution.py
Author: Keith Tauscher
Date: 23 Sep 2018

Description: Example script illustrating use of WindowedDistribution alongside
             UniformDistribution and GaussianDistribution classes to emulate a
             truncated Gaussian distribution (this is for demonstration
             purposes only; if you actually want to use a truncated Gaussian
             distribution, it is more efficient to use the
             TruncatedGaussianDistribution class).
"""
from __future__ import division
import time
import numpy as np
import matplotlib.pyplot as pl
from distpy import UniformDistribution, GaussianDistribution,\
    WindowedDistribution, TruncatedGaussianDistribution

ndraw = int(1e5)
nbins = 50

background_distribution = GaussianDistribution(-10, 4)
foreground_distribution = UniformDistribution(-12, -7)
distribution =\
    WindowedDistribution(background_distribution, foreground_distribution)
equivalent_truncated_gaussian_distribution =\
    TruncatedGaussianDistribution(-10, 4, -12, -7)

start_time = time.time()
sample = distribution.draw(ndraw)
end_time = time.time()
duration = end_time - start_time
print(("It took {0:.5f} s to draw {1:d} points from a WindowedDistribution " +\
    "with a GaussianDistribution background_distribution and a " +\
    "UniformDistribution foreground_distribution.").format(duration, ndraw))

fig = pl.figure(figsize=(12,9))
ax = fig.add_subplot(111)
ax.hist(sample, histtype='step', bins=nbins, color='k', density=True)
num_x_values = 100
x_values =\
    np.linspace(*([element for element in ax.get_xlim()] + [num_x_values]))
equivalent_truncated_gaussian_distribution.plot(x_values, ax=ax, show=False,\
    color='b', label='TruncatedGaussianDistribution')
scale_factor = 1 / distribution.approximate_acceptance_fraction(ndraw)
distribution.plot(x_values, scale_factor=scale_factor, ax=ax, show=False,\
    color='g', label='scaled WindowedDistribution')
ax.legend()

pl.show()

