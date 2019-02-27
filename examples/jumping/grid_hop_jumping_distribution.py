"""
File: examples/jumping/grid_hop_jumping_distribution.py
Author: Keith Tauscher
Date: 22 Sep 2018

Description: Script showing an example of the use of the
             GridHopJumpingdistribution class, which only takes jumps through
             discrete space which have taxi cab distance 0 or 1 and does not
             favor any given direction.
"""
import os, time
import numpy as np
import matplotlib.pyplot as pl
from distpy import GridHopJumpingDistribution,\
    load_jumping_distribution_from_hdf5_file

ndim = 2
ndraw = int(1e4)
fontsize = 24

minimum = 1
maximum = 5
minima = (minimum,) * ndim
maxima = (maximum,) * ndim
jumping_probability = 0.5
distribution = GridHopJumpingDistribution(ndim=ndim,\
    jumping_probability=jumping_probability, minima=minima, maxima=maxima)

file_name = 'TESTINGGRIDHOPJUMPINGDISTRIBUTIONDELETETHISIFYOUSEEIT.hdf5'
try:
    distribution.save(file_name)
    assert(distribution == load_jumping_distribution_from_hdf5_file(file_name))
except:
    os.remove(file_name)
    raise
else:
    os.remove(file_name)

sources = np.array([(1, 3), (4, 4), (5, 5)])

start_time = time.time()
samples = [distribution.draw(source, shape=ndraw) for source in sources]
end_time = time.time()
duration = end_time - start_time
print(("It took {0:.5f} s to draw {1:d} samples of size {2:d} from a " +\
    "GridHopJumpingDistribution in {3:d} dimensions.").format(duration,\
    len(sources), ndraw, ndim))
num_samples = len(samples)

fig = pl.figure(figsize=(27,9))
bins = 2 * (np.arange(maximum - minimum + 2) + 0.5,)
lim = (minimum - 0.5, maximum + 0.5)
for (sample_index, sample) in enumerate(samples):
    ax = fig.add_subplot(1, num_samples, 1 + sample_index)
    ax.hist2d(sample[:,0], sample[:,1], cmap='viridis', bins=bins)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel('x', size=fontsize)
    ax.set_ylabel('y', size=fontsize)
    ax.tick_params(labelsize=fontsize, width=2.5, length=7.5, which='major')
    ax.tick_params(labelsize=fontsize, width=1.5, length=4.5, which='minor')

pl.show()

