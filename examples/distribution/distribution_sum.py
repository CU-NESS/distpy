"""
File: examples/distribution/distribution_sum.py
Author: Keith Tauscher
Date: Oct 15 2019

Description: Example script showing how to generate and use a DistributionSum
             object, which allows for multiple unrelated distributions to be
             summed together consistently.
"""
import os, time
import numpy as np
import matplotlib.pyplot as pl
from distpy import GaussianDistribution, UniformDistribution,\
    DistributionSum, load_distribution_from_hdf5_file

ndraw = int(1e5)

means = (-3, 3)
variances = (1 / 9., 1.)
distributions = [GaussianDistribution(mean, variance)\
    for (mean, variance) in zip(means, variances)] +\
    [UniformDistribution(-5, 5)]
weights = np.array([1., 3., 1.])

distribution = DistributionSum(distributions, weights)
hdf5_file_name = 'testing_distribution_sum_class_DELETE_THIS.hdf5'

try:
    distribution.save(hdf5_file_name)
    assert(distribution == load_distribution_from_hdf5_file(hdf5_file_name))
    assert(distribution == DistributionSum.load(hdf5_file_name,\
        *map(lambda x : x.__class__, distributions)))
except:
    os.remove(hdf5_file_name)
    raise
else:
    os.remove(hdf5_file_name)

start_time = time.time()
draws = distribution.draw(ndraw)
end_time = time.time()
duration = end_time - start_time
print(("It took {0:.5f} s to draw a sample of size {1:d} from a " +\
    "DistributionSum of a UniformDistribution and a " +\
    "GaussianDistribution.").format(duration, ndraw))
print('Sample mean was {0:.3g}, while expected mean was {1:.3g}.'.format(\
    np.mean(draws), distribution.mean))
print(('Sample standard deviation was {0:.3g}, while expected standard ' +\
    'deviation was {1:.3g}.').format(np.std(draws),\
    distribution.standard_deviation))

xs = np.linspace(-8, 8, 1000)
ys = np.array([np.exp(distribution.log_value(x)) for x in xs])

fig = pl.figure()
ax = fig.add_subplot(111)
ax.hist(draws, histtype='stepfilled', density=True, label='sampled', bins=xs)
distribution.plot(xs, ax=ax, show=True, label='e^(log_value)')

