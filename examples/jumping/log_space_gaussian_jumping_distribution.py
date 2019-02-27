"""
File: examples/jumping/log_space_gaussian_jumping_distribution.py
Author: Keith Tauscher
Date: 24 Dec 2017

Description: Example showing how to use the JumpingDistributionSet to define
             JumpingDistribution objects in transformed spaces with respect to
             those that define the variable(s) they describe.
"""
import os, time
import matplotlib.pyplot as pl
from distpy import GaussianJumpingDistribution, JumpingDistributionSet

jumping_distribution = GaussianJumpingDistribution(1.)
distribution_set = JumpingDistributionSet()
distribution_set.add_distribution(jumping_distribution, 'x', 'log')

try:
    file_name = 'TEMPORARY_TEST_DELETE_IF_EXISTS.hdf5'
    distribution_set.save(file_name)
    loaded_distribution_set = JumpingDistributionSet.load(file_name)
    assert loaded_distribution_set == distribution_set
except:
    os.remove(file_name)
    raise
else:
    os.remove(file_name)

sample_size = int(1e5)

sources = [(10 ** i) for i in range(-1, 1)]

t0 = time.time()
samples = [distribution_set.draw({'x': source}, shape=sample_size)['x']\
    for source in sources]
t1 = time.time()
print(("It took {0:.5f} s to draw {1:d} samples from a " +\
    "GaussianJumpingDistribution in log space.").format(t1 - t0, sample_size))

for (isample, sample) in enumerate(samples):
    pl.hist(sample, bins=100, histtype='step', color='C{}'.format(isample))

pl.show()

