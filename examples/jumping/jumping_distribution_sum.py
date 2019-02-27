"""
File: examples/jumping/jumping_distribution_sum.py
Author: Keith Tauscher
Date: 26 Feb 2019

Description: Example script showing the creation and use of a
             JumpingDistributionSum, which allows for multiple
             JumpingDistribution classes to be used according to a set of
             discrete probabilities.
"""
import os, time
import numpy as np
import matplotlib.pyplot as pl
from distpy import GaussianJumpingDistribution, UniformJumpingDistribution,\
    JumpingDistributionSum, load_jumping_distribution_from_hdf5_file

ndraw = int(1e5)
source = 1.

variances = (1 / 9., 1.)
jumping_distributions = [GaussianJumpingDistribution(1.),\
    UniformJumpingDistribution(1 / 3.)]
weights = np.array([1., 2.])

jumping_distribution = JumpingDistributionSum(jumping_distributions, weights)
hdf5_file_name = 'testing_jumping_distribution_sum_class_DELETE_THIS.hdf5'

try:
    jumping_distribution.save(hdf5_file_name)
    assert(jumping_distribution ==\
        load_jumping_distribution_from_hdf5_file(hdf5_file_name))
    assert(jumping_distribution == JumpingDistributionSum.load(hdf5_file_name,\
        *map(lambda x : x.__class__, jumping_distributions)))
except:
    os.remove(hdf5_file_name)
    raise
else:
    os.remove(hdf5_file_name)

start_time = time.time()
draws = jumping_distribution.draw(source, shape=ndraw)
end_time = time.time()
duration = end_time - start_time
print(("It took {0:.5f} s to draw a sample of size {1:d} from a " +\
    "JumpingDistributionSum containing a GaussianJumpingDistribution and a " +\
    "UniformJumpingDistribution.").format(duration, ndraw))

xs = np.linspace(source - 4, source + 4, 1000)
ys = np.array([np.exp(jumping_distribution.log_value(source, x)) for x in xs])

pl.hist(draws, histtype='stepfilled', density=True, label='sampled', bins=xs)
pl.plot(xs, ys, label='exp(log_value)')
pl.legend()
pl.show()

