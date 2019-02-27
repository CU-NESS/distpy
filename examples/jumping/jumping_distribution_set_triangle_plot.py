"""
File: examples/distribution/jumping_distribution_set_triangle_plot.py
Author: Keith Tauscher
Date: 22 Nov 2018

Description: Example script showing how to quickly create a triangle plot from
             a JumpingDistributionSet object.
"""
import time
import numpy as np
import matplotlib.pyplot as pl
from distpy import TransformList, GaussianJumpingDistribution,\
    JumpingDistributionSet

ndraw = int(1e5)
nbins = 100
contour_confidence_levels = 0.997
number_of_sigma = np.sqrt((-2) * np.log(1 - contour_confidence_levels))
source = {'x': 1, 'y': 2, 'z': 1}
parameters = ['x', 'y', 'z']
mean = np.array([source[parameter] for parameter in parameters])
covariance = np.array([[2, 0.8, 0], [0.8, 0.5, 0], [0, 0, 4]])
jumping_distribution = GaussianJumpingDistribution(covariance)
transform_list = TransformList(None, None, 'log10')
transformed_mean = transform_list(mean)
jumping_distribution_set = JumpingDistributionSet([\
    (jumping_distribution, parameters, transform_list)])
start_time = time.time()
jumping_distribution_set.triangle_plot(source, ndraw, nbins=nbins, show=True,\
    in_transformed_space=True, reference_value_mean=transformed_mean,\
    reference_value_covariance=(number_of_sigma**2)*covariance,\
    contour_confidence_levels=contour_confidence_levels,\
    parameter_renamer=(lambda x: '${!s}$'.format(x)))
end_time = time.time()
duration = end_time - start_time
print(("It took {0:.5f} s to generate a triangle plot with {1:d} samples " +\
    "from a JumpingDistributionSet consisting of a 3D Gaussian with one " +\
    "variable in log space and two untransformed variables.").format(duration,\
    ndraw))
pl.show()

