"""
File: examples/distribution/generalized_pareto_distribution.py
Author: Keith Tauscher
Date: 28 May 2018

Description: Example of using the GeneralizedParetoDistribution.
"""
import os, time
import numpy as np
import matplotlib.pyplot as pl
from distpy import GeneralizedParetoDistribution

nsigma_for_x_axis = 3

shape = 4
if shape <= 2:
    raise ValueError("shape less than or equal to 2.")
sample_size = int(1e5)
maximum =\
    (1 + (nsigma_for_x_axis * np.sqrt(shape / (shape - 2)))) / (shape - 1)

distribution = GeneralizedParetoDistribution(shape)
hdf5_file_name = 'TEST_DELETE_THIS.hdf5'
distribution.save(hdf5_file_name)
try:
    assert (distribution == GeneralizedParetoDistribution.load(hdf5_file_name))
except:
    os.remove(hdf5_file_name)
    raise
else:
    os.remove(hdf5_file_name)
assert distribution.numparams == 1
t0 = time.time()
sample = distribution.draw(sample_size)
print(('It took {0:.5f} s to draw {1} points from a generalized pareto ' +\
    'distribution.').format(time.time() - t0, sample_size))
pl.figure()
nbins = 100
bins = np.linspace(0, maximum, nbins)
pl.hist(sample, bins=bins, histtype='step', color='b', linewidth=2,\
    normed=True, label='sampled')
pl.plot(bins, list(map((lambda x : np.exp(distribution.log_value(x))), bins)),\
    linewidth=2, color='r', label='e^(log_value)')
ylim = pl.ylim()
for xval in distribution.central_confidence_interval(0.5):
    pl.plot(2 * [xval], ylim, color='k')
pl.ylim(ylim)
pl.title(distribution.to_string(), size='xx-large')
pl.xlabel('Value', size='xx-large')
pl.ylabel('PDF', size='xx-large')
pl.tick_params(labelsize='xx-large', width=2, length=6)
pl.legend(fontsize='xx-large', loc='lower center')
pl.xlim((0, maximum))
pl.show()



