"""
File: examples/gridded_distribution.py
Author: Keith Tauscher
Date: 7 Aug 2017

Description: Example of using the GriddedDistribution class to use pdf's
             defined on grids.
"""
import time
import numpy as np
import matplotlib.pyplot as pl
import matplotlib.cm as cm
from distpy import GriddedDistribution

def_cm = cm.bone
sample_size = int(1e5)

def pdf_func(x,y):
    if (y > (10. - ((x ** 2) / 5.))) and\
       (y < ((4. * x) + 30.)) and\
       (y < (-4. * x) + 30.) and\
       (x >= -10.) and (x <= 10.):
        return np.exp(-((x ** 2) + ((y - 10.) ** 2)) / 200.)
    else:
        return 0.

xs = np.arange(-20., 20.1, 0.1)
ys = np.arange(-10., 30., 0.1)
pdf = np.ndarray((len(xs), len(ys)))
for ix in range(len(xs)):
    for iy in range(len(ys)):
        pdf[ix,iy] = pdf_func(xs[ix], ys[iy])
distribution = GriddedDistribution([xs, ys], pdf=pdf)
t0 = time.time()
sample = distribution.draw(sample_size)
print(("It took {0:.5f} s to draw {1} points from a user-defined " +\
    "distribution with {2} pixels.").format(time.time() - t0, sample_size,\
    len(xs) * len(ys)))
sampled_xs = [sample[i][0] for i in range(sample_size)]
sampled_ys = [sample[i][1] for i in range(sample_size)]
pl.figure()
pl.hist2d(sampled_xs, sampled_ys, bins=100, cmap=def_cm)
pl.title('Points sampled from a user-defined distribution',\
    size='xx-large')
pl.xlabel('x', size='xx-large')
pl.ylabel('y', size='xx-large')
pl.tick_params(labelsize='xx-large', width=2, length=6)
pdf_from_log_value = np.ndarray((len(ys), len(xs)))
Xs, Ys = np.meshgrid(xs, ys)
for ix in range(len(xs)):
    for iy in range(len(ys)):
        pdf_from_log_value[iy,ix] =\
            np.exp(distribution.log_value([Xs[iy,ix], Ys[iy,ix]]))
pl.figure()
pl.imshow(pdf_from_log_value / np.max(pdf_from_log_value), origin='lower',\
    cmap=def_cm, extent=[-20., 20., -10., 30.])
pl.gca().set_aspect('equal', adjustable='box')
pl.title('e^(log_value) for same distribution as previous sample',\
    size='xx-large')
pl.xlabel('x', size='xx-large')
pl.ylabel('y', size='xx-large')
pl.tick_params(labelsize='xx-large', width=2, length=6)
pl.show()

