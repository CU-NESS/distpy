"""
File: examples/distribution/distribution_set_triangle_plot.py
Author: Keith Tauscher
Date: 22 Nov 2018

Description: Example script showing how to quickly create a triangle plot from
             a DistributionSet object.
"""
import time
import numpy as np
import matplotlib.pyplot as pl
from distpy import TransformList, GaussianDistribution, DistributionSet

ndraw = int(1e5)
nbins = 100
contour_confidence_levels = 0.997
parameters = ['x', 'y', 'z']
mean = np.array([1, 2, 0])
covariance = np.array([[2, 0.8, 0], [0.8, 0.5, 0], [0, 0, 4]])
distribution = GaussianDistribution(mean, covariance)
transform_list = TransformList(None, None, 'log10')
distribution_set =\
    DistributionSet([(distribution, parameters, transform_list)])
start_time = time.time()
distribution_set.triangle_plot(ndraw, nbins=nbins, show=False,\
    in_transformed_space=True, reference_value_mean=mean,\
    reference_value_covariance=covariance,\
    contour_confidence_levels=contour_confidence_levels,\
    parameter_renamer=(lambda x: '${!s}$'.format(x)))
end_time = time.time()
duration = end_time - start_time
print(("It took {0:.5f} s to assemble a triangle plot for a 3D " +\
    "GaussianDistribution with two non-transformed variables and one " +\
    "log-transformed variables with a sample of size {1:d}.").format(duration,\
    ndraw))
pl.show()

