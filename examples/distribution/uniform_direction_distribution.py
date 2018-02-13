"""
File: examples/distribution/uniform_direction_distribution.py
Author: Keith Tauscher
Date: 10 Aug 2017

Description: File containing example of using the UniformDirectionDistribution
             class.
"""
import os
import numpy as np
import matplotlib.pyplot as pl
import healpy as hp
from distpy import UniformDirectionDistribution

nsiden = 5
nside = 2 ** nsiden
npix = hp.pixelfunc.nside2npix(nside)

pointing_center = (0, 0)
distribution = UniformDirectionDistribution(pointing_center=pointing_center,\
    high_theta=np.radians(90))
hdf5_file_name = 'TEST_DELETE_THIS.hdf5'
distribution.save(hdf5_file_name)
try:
    distribution == UniformDirectionDistribution.load(hdf5_file_name)
except:
    os.remove(hdf5_file_name)
    raise
else:
    os.remove(hdf5_file_name)
ndraws = int(1e6)
draws = distribution.draw(ndraws)
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

print(distribution.to_string())

pl.figure()
pl.hist(lons, bins=10)
pl.title('Longitude distribution')
pl.figure()
pl.hist(lats, bins=10)
pl.title('Latitude distribution')

pl.show()
