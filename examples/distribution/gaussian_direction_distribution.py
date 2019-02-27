"""
File: examples/distribution/gaussian_direction_distribution.py
Author: Keith Tauscher
Date: 10 Aug 2017

Description: File containing example of using the GaussianDirectionDistribution
             class.
"""
import os, time
import numpy as np
import matplotlib.pyplot as pl
import healpy as hp
from distpy import GaussianDirectionDistribution

nsiden = 5
nside = 2 ** nsiden
npix = hp.pixelfunc.nside2npix(nside)

pointing_center = (0, 0)
distribution = GaussianDirectionDistribution(pointing_center=pointing_center,\
    sigma=10, degrees=True)

hdf5_file_name = 'TEST_DELETE_THIS.hdf5'
distribution.save(hdf5_file_name)
try:
    assert(distribution == GaussianDirectionDistribution.load(hdf5_file_name))
    assert(distribution.to_string() == 'GaussianDirection((0, 0), 0.175)')
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
print(("It took {0:.5f} s to draw {1:d} samples from a " +\
    "GaussianDirectionDistribution.").format(duration, ndraws))
lons = draws[:,1]
lats = draws[:,0]
alpha = 0.1

(data_lons, data_lats) =\
    hp.pixelfunc.pix2ang(nside, np.arange(npix), lonlat=True)
log_values = np.array([distribution.log_value((data_lats[i], data_lons[i]))\
                                                         for i in range(npix)])
values = np.exp(log_values)
hp.mollview(values, title='Distribution')

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

