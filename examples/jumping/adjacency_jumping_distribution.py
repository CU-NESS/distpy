"""
File: examples/jumping/adjacency_jumping_distribution.py
Author: Keith Tauscher
Date: 22 Dec 2017

Description: Example of using the AdjacencyJumpingDistribution class to
             represent 1D integer random variates.
"""
import os, time
import numpy as np
import matplotlib.pyplot as pl
from matplotlib.ticker import MultipleLocator
from distpy import AdjacencyJumpingDistribution,\
    load_jumping_distribution_from_hdf5_file

sample_size = int(1e5)
source1 = 1
source2 = 2
source3 = 11
source4 = 21
source5 = 20
jumping_probability = 0.4
(minimum, maximum) = (1, 21)
distribution = AdjacencyJumpingDistribution(\
    jumping_probability=jumping_probability, minimum=minimum, maximum=maximum)
assert distribution.numparams == 1
try:
    file_name = 'TEMPORARY_TEST_DELETE_THIS_IF_IT_EXISTS.hdf5'
    distribution.save(file_name)
    loaded_distribution = load_jumping_distribution_from_hdf5_file(file_name)
    assert loaded_distribution == distribution
except:
    os.remove(file_name)
    raise
else:
    os.remove(file_name)

t0 = time.time()
sample1 = distribution.draw(source1, sample_size)
sample2 = distribution.draw(source2, sample_size)
sample3 = distribution.draw(source3, sample_size)
sample4 = distribution.draw(source4, sample_size)
sample5 = distribution.draw(source5, sample_size)
print(('It took {0:.5f} s for five samples of size {1} to be drawn from a ' +\
    'adjacency jumping distribution.').format(time.time() - t0, sample_size))
pl.figure()
bins = np.arange(minimum - 0.5, maximum + 1.5)
pl.hist(sample1, bins=bins, histtype='step', color='C0', linewidth=2,\
    label='source={} (sampled)'.format(source1), density=True)
pl.hist(sample2, bins=bins, histtype='step', color='C1', linewidth=2,\
    label='source={} (sampled)'.format(source2), density=True)
pl.hist(sample3, bins=bins, histtype='step', color='C2', linewidth=2,\
    label='source={} (sampled)'.format(source3), density=True)
pl.hist(sample4, bins=bins, histtype='step', color='C3', linewidth=2,\
    label='source={} (sampled)'.format(source4), density=True)
pl.hist(sample5, bins=bins, histtype='step', color='C4', linewidth=2,\
    label='source={} (sampled)'.format(source5), density=True)
xs = np.arange(minimum, maximum + 1)
y1s = list(map((lambda x : np.exp(distribution.log_value(source1, x))), xs))
y2s = list(map((lambda x : np.exp(distribution.log_value(source2, x))), xs))
y3s = list(map((lambda x : np.exp(distribution.log_value(source3, x))), xs))
y4s = list(map((lambda x : np.exp(distribution.log_value(source4, x))), xs))
y5s = list(map((lambda x : np.exp(distribution.log_value(source5, x))), xs))
pl.scatter(xs, y1s, linewidth=2, color='C0', linestyle='--',\
    label='source={} (e^(log_prior))'.format(source1))
pl.scatter(xs, y2s, linewidth=2, color='C1', linestyle='--',\
    label='source={} (e^(log_prior))'.format(source2))
pl.scatter(xs, y3s, linewidth=2, color='C2', linestyle='--',\
    label='source={} (e^(log_prior))'.format(source3))
pl.scatter(xs, y4s, linewidth=2, color='C3', linestyle='--',\
    label='source={} (e^(log_prior))'.format(source4))
pl.scatter(xs, y5s, linewidth=2, color='C4', linestyle='--',\
    label='source={} (e^(log_prior))'.format(source5))
pl.xlim((minimum, maximum))
pl.ylim((0, pl.ylim()[1] * 1.25))
pl.gca().xaxis.set_major_locator(MultipleLocator(1))
pl.title(('Adjacency distribution with jumping_probability={0}, min={1}, ' +\
    'and max={2}').format(jumping_probability, minimum, maximum),\
    size='xx-large')
pl.xlabel('Value', size='xx-large')
pl.ylabel('PDF', size='xx-large')
pl.tick_params(labelsize='xx-large', width=2, length=6)
pl.legend(fontsize='xx-large', ncol=2, loc='upper center')
pl.show()

