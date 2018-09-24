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

sample = distribution.draw(ndraw)

fig = pl.figure(figsize=(12,9))
ax = fig.add_subplot(111)

ax.hist(sample, histtype='step', bins=nbins, color='k', normed=True)
num_x_values = 100
x_values =\
    np.linspace(*([element for element in ax.get_xlim()] + [num_x_values]))
proportional_y_values =\
    np.exp([distribution.log_value(x_value) for x_value in x_values])
equal_y_values =\
    np.exp([equivalent_truncated_gaussian_distribution.log_value(x_value)\
    for x_value in x_values])
ax.plot(x_values, equal_y_values, label='TruncatedGaussianDistribution',\
    color='b')
ax.plot(x_values, proportional_y_values, label='WindowedDistribution',\
    color='g')
ax.legend()

pl.show()

