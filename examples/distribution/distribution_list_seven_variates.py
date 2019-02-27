"""
File: examples/distribution/distribution_list_seven_variates.py
Author: Keith Tauscher
Date: 23 Sep 2018

Description: File containing example of a seven-variate DistributionList.
"""
import os, time
import numpy as np
import matplotlib.pyplot as pl
import matplotlib.cm as cm
from distpy import DistributionList, GaussianDistribution, BetaDistribution,\
    ParallelepipedDistribution, load_distribution_from_hdf5_file

sample_size = int(1e6)

distribution_list = DistributionList()
distribution_list.add_distribution(GaussianDistribution(5., 4.))
distribution_list.add_distribution(BetaDistribution(18., 3.))
distribution_list.add_distribution(GaussianDistribution([5., 2., 9.],\
    [[4., 1., 1.], [1., 4., 1.], [1., 1., 4.]]))
distribution_list.add_distribution(ParallelepipedDistribution([69., 156.],\
    [[1.,-1.], [1.,1.]], [1., 1.]))
hdf5_file_name = 'TEST_DELETE_THIS.hdf5'
distribution_list.save(hdf5_file_name)
try:
    assert(distribution_list == DistributionList.load(hdf5_file_name,\
        GaussianDistribution, BetaDistribution, GaussianDistribution,\
        ParallelepipedDistribution))
    assert(distribution_list ==\
        load_distribution_from_hdf5_file(hdf5_file_name))
except:
    os.remove(hdf5_file_name)
    raise
else:
    os.remove(hdf5_file_name)
t0 = time.time()
sample = distribution_list.draw(sample_size)
print(('It took {0:.5f} s to draw {1} points from a mixed distribution ' +\
    'with 7 parameters, in groups of 3, 2, 1, and 1.').format(\
    time.time() - t0, sample_size))
pl.figure()
pl.hist2d(sample[:,1], sample[:,6], bins=100, cmap=cm.bone)
pl.title('PriorSet seven-parameter test. x should be in [0,1] and y should ' +\
         'be around 156', size='xx-large')
pl.xlabel('x', size='xx-large')
pl.ylabel('y', size='xx-large')
pl.tick_params(labelsize='xx-large', width=2, length=6)
pl.show()

