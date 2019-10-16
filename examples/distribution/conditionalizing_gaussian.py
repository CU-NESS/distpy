"""
File: examples/distribution/conditionalizing_gaussian.py
Author: Keith Tauscher
Date: 1 Mar 2018

Description: File showing how to conditionalize and marginlize
             GaussianDistribution objects.
"""
import numpy as np
from distpy import GaussianDistribution

# correlation and known_y are the only free parameters here.
correlation = -0.8
known_y = 0.9

y_index = 1
# y_index could also be a list, numpy.ndarray or slice
mean = np.zeros(2)
covariance = np.array([[1, correlation], [correlation, 1]])
expected_conditionalized_variance = (1 - (correlation ** 2))
expected_conditionalized_mean = correlation * (known_y - mean[1])
joint = GaussianDistribution(mean, covariance)
conditionalized = joint.conditionalize(y_index, known_y)
conditionalized_variance = conditionalized.variance
conditionalized_mean = conditionalized.mean
marginalized = joint[0]
expected_marginalized_variance = 1
expected_marginalized_mean = 0
marginalized_mean = marginalized.mean
marginalized_variance = marginalized.variance

actual = [conditionalized_variance, conditionalized_mean] +\
    [marginalized_variance, marginalized_mean]
expected =\
    [expected_conditionalized_variance, expected_conditionalized_mean] +\
    [expected_marginalized_variance, expected_marginalized_mean]

assert(np.allclose(actual, expected, rtol=0, atol=1e-6))

