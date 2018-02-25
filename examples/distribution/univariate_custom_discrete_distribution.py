"""
File: examples/distribution/univariate_custom_discrete_distribution.py
Author: Keith Tauscher
Date: 24 Feb 2018

Description: Example showing how to use the CustomDiscreteDistribution class to
             represent a 1D discrete distribution with a uniform probability
             mass function.
"""
import os, time
import numpy as np
import matplotlib.pyplot as pl
from distpy import CustomDiscreteDistribution, load_distribution_from_hdf5_file

fontsize = 24
nvalues = 20
x_values = np.arange(nvalues) + 1
bins = np.concatenate([x_values - 0.5, [x_values[-1] + 0.5]])
xlim = (x_values[0] - 1, x_values[-1] + 1)
pmf_values = np.ones(nvalues)
distribution = CustomDiscreteDistribution(x_values, pmf_values)
hdf5_file_name = 'TEST_DELETE_THIS.hdf5'
distribution.save(hdf5_file_name)
try:
    assert distribution == load_distribution_from_hdf5_file(hdf5_file_name)
except:
    os.remove(hdf5_file_name)
    raise
else:
    os.remove(hdf5_file_name)

ndraw = int(1e4)
t0 = time.time()
draws = distribution.draw(ndraw)
t1 = time.time()
print(("It took {0:.5g} s to draw {1:d} samples from a 1D custom discrete " +\
    "distribution with {2:d} possible values.").format(t1 - t0, ndraw,\
    nvalues))

log_values = np.array([distribution.log_value(x_val) for x_val in x_values])

pl.figure(figsize=(12, 10))
pl.hist(draws, bins=bins, color='r', histtype='step', normed=True)
pl.scatter(x_values, np.exp(log_values), color='r')
pl.xlim(xlim)
pl.tick_params(width=2.5, length=7.5, labelsize=fontsize)
pl.xlabel('x', size=fontsize)
pl.ylabel('Probability', size=fontsize)
pl.title('Observed and expected cumulative mass function', size=fontsize)
pl.show()

