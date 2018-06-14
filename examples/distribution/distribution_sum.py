import os
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

draws = distribution.draw(ndraw)

xs = np.linspace(-8, 8, 1000)
ys = np.array([np.exp(distribution.log_value(x)) for x in xs])

pl.hist(draws, histtype='stepfilled', normed=True, label='sampled', bins=xs)
pl.plot(xs, ys, label='exp(log_value)')
pl.legend()
pl.show()

