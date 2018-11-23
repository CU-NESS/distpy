"""
File: examples/distribution/distribution_set_triangle_plot.py
Author: Keith Tauscher
Date: 22 Nov 2018

Description: Example script showing how to quickly create a triangle plot from
             a DistributionSet object.
"""
import numpy as np
from distpy import TransformList, GaussianDistribution, DistributionSet

ndraw = int(1e5)
nbins = 100
contour_confidence_levels = 0.997
number_of_sigma = np.sqrt((-2) * np.log(1 - contour_confidence_levels))
parameters = ['x', 'y', 'z']
mean = np.array([1, 2, 0])
covariance = np.array([[2, 0.8, 0], [0.8, 0.5, 0], [0, 0, 4]])
distribution = GaussianDistribution(mean, covariance)
transform_list = TransformList(None, None, 'log10')
distribution_set =\
    DistributionSet([(distribution, parameters, transform_list)])
distribution_set.triangle_plot(ndraw, nbins=nbins, show=True,\
    in_transformed_space=True, reference_value_mean=mean,\
    reference_value_covariance=(number_of_sigma**2)*covariance,\
    contour_confidence_levels=contour_confidence_levels,\
    parameter_renamer=(lambda x: '${!s}$'.format(x)))
