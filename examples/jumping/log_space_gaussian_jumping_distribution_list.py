"""
File: examples/jumping/log_space_gaussian_jumping_distribution.py
Author: Keith Tauscher
Date: 24 Dec 2017

Description: Example showing how to use the JumpingDistributionSet to define
             JumpingDistribution objects in transformed spaces with respect to
             those that define the variable(s) they describe.
"""
import os, time
import numpy as np
import matplotlib.pyplot as pl
from distpy import GaussianJumpingDistribution, JumpingDistributionList

jumping_distribution = GaussianJumpingDistribution(0.25)
jumping_distribution_list = JumpingDistributionList()
jumping_distribution_list.add_distribution(jumping_distribution, 'log')

try:
    file_name = 'TEMPORARY_TEST_DELETE_IF_EXISTS.hdf5'
    jumping_distribution_list.save(file_name)
    loaded_jumping_distribution_list = JumpingDistributionList.load(file_name)
    assert loaded_jumping_distribution_list == jumping_distribution_list
except:
    os.remove(file_name)
    raise
else:
    os.remove(file_name)

sample_size = int(1e5)

sources = [(10 ** i) for i in [0.9, 1.4]]

t0 = time.time()
samples = [jumping_distribution_list.draw(source, shape=sample_size)\
    for source in sources]
t1 = time.time()
print(("It took {0:.5f} s to draw {1:d} samples from a " +\
    "GaussianJumpingDistribution in log space.").format(t1 - t0, sample_size))

fig = pl.figure(figsize=(12,9))
ax = fig.add_subplot(111)
for (index, (source, sample)) in enumerate(zip(sources, samples)):
    ax.hist(sample, bins=1000, histtype='step', color='C{}'.format(index),\
        density=True, alpha=0.25)
    jumping_distribution_list.plot(source,\
        np.power(10, np.linspace(-2.1, 4.4, 1000)), scale_factor=1, xlabel='x',\
        ylabel='PDF', fontsize=24, ax=ax, show=False,\
        color='C{}'.format(index), linestyle='--')

ax.set_xscale('log')

#ax.set_ylim((0, 0.15))
ax.set_yscale('log')
ax.set_ylim((1e-6, 2e-1))

pl.show()

