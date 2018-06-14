import os
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

draws = jumping_distribution.draw(source, shape=ndraw)

xs = np.linspace(source - 4, source + 4, 1000)
ys = np.array([np.exp(jumping_distribution.log_value(source, x)) for x in xs])

pl.hist(draws, histtype='stepfilled', normed=True, label='sampled', bins=xs)
pl.plot(xs, ys, label='exp(log_value)')
pl.legend()
pl.show()

