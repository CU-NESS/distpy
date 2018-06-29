import numpy as np
import matplotlib.pyplot as pl
from distpy import GaussianDistribution, DistributionSet,\
    DistributionHarmonizer

fontsize = 24

known_distribution_set =\
    DistributionSet([(GaussianDistribution(0, 1), 'x', None)])
def solver(dictionary):
    return {'y': 1 + dictionary['x']}
ndraw = 10000

distribution_harmonizer =\
    DistributionHarmonizer(known_distribution_set, solver, ndraw)
full_distribution_set = distribution_harmonizer.full_distribution_set

sample = full_distribution_set.draw(ndraw)
sample = np.array([sample['x'], sample['y']])

fig = pl.figure(figsize=(12, 9))
ax = fig.add_subplot(111)
ax.hist2d(sample[0], sample[1],\
    bins=(np.linspace(-3, 3, 10), np.linspace(-2, 4, 10)))
ax.set_xlabel('x', size=fontsize)
ax.set_ylabel('y', size=fontsize)
ax.tick_params(labelsize=fontsize, width=2.5, length=7.5, which='major')
ax.tick_params(labelsize=fontsize, width=1.5, length=4.5, which='minor')

pl.show()

