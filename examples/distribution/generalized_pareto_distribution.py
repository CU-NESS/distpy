"""
File: examples/distribution/generalized_pareto_distribution.py
Author: Keith Tauscher
Date: Oct 15 2019

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
sample_size = int(1e6)
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
print('Sample mean was {0:.3g}, while expected mean was {1:.3g}.'.format(\
    np.mean(sample), distribution.mean))
print(('Sample standard deviation was {0:.3g}, while expected standard ' +\
    'deviation was {1:.3g}.').format(np.std(sample),\
    distribution.standard_deviation))
fig = pl.figure()
ax = fig.add_subplot(111)
nbins = 100
bins = np.linspace(0, maximum, nbins)
ax.hist(sample, bins=bins, histtype='step', color='b', linewidth=2,\
    density=True, label='sampled')
distribution.plot(bins, ax=ax, show=False, linewidth=2, color='r',\
    label='e^(log_value)')
ylim = ax.get_ylim()
for xval in distribution.central_confidence_interval(0.5):
    ax.plot(2 * [xval], ylim, color='k')
ax.set_ylim(ylim)
ax.set_title(distribution.to_string(), size='xx-large')
ax.set_xlabel('Value', size='xx-large')
ax.set_ylabel('PDF', size='xx-large')
ax.tick_params(labelsize='xx-large', width=2, length=6)
ax.legend(fontsize='xx-large', loc='lower center')
ax.set_xlim((0, maximum))
pl.show()



