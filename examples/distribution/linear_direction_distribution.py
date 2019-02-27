"""
File: examples/distribution/gaussian_direction_distribution.py
Author: Keith Tauscher
Date: 10 Aug 2017

Description: File containing example of using the GaussianDirectionDistribution
             class.
"""
from __future__ import division
import os, time
import numpy as np
import matplotlib.pyplot as pl
import healpy as hp
from distpy import GaussianDistribution, LinearDirectionDistribution

nsiden = 6
nside = 2 ** nsiden
npix = hp.pixelfunc.nside2npix(nside)

central_pointing = (0, 0)
phase_delayed_pointing = (45, 45)
angle_distribution = GaussianDistribution(0, (np.pi / 4) ** 2)
distribution = LinearDirectionDistribution(central_pointing,\
    phase_delayed_pointing, angle_distribution)

hdf5_file_name = 'TEST_DELETE_THIS.hdf5'
distribution.save(hdf5_file_name)
try:
    assert(distribution == LinearDirectionDistribution.load(hdf5_file_name,\
        type(angle_distribution)))
except:
    os.remove(hdf5_file_name)
    raise
else:
    os.remove(hdf5_file_name)

ndraws = int(1e6)
start_time = time.time()
draws = distribution.draw(ndraws)
end_time = time.time()
duration = end_time - start_time
print(("It took {0:.5f} s to draw a sample of size {1:d} from a " +\
    "LinearDirectiondistribution.").format(duration, ndraws))
lons = draws[:,1]
lats = draws[:,0]

pixels_of_draws = hp.pixelfunc.ang2pix(nside, lons, lats, lonlat=True)
histogram = np.zeros(npix, dtype=int)
unique, counts = np.unique(pixels_of_draws, return_counts=True)
for ipixel in range(len(unique)):
    histogram[unique[ipixel]] = counts[ipixel]
histogram = histogram / (4 * np.pi * np.mean(histogram))
hp.mollview(histogram, title='Histogram of draws')

pl.figure()
pl.hist(lons, bins=10)
pl.title('Longitude distribution')
pl.figure()
pl.hist(lats, bins=10)
pl.title('Latitude distribution')

pl.show()

