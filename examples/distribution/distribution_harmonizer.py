"""
File: examples/distribution/distribution_harmonizer.py
Author: Keith Tauscher
Date: 26 Feb 2019

Description: Example script showing how to use a DistributionHarmonizer object,
             which creates a consistent N-D distribution out of a M-D
             marginal distribution and a conditional_solver which computes the
             conditional distribution of the remaining (N-M) parameters.
"""
import time
import numpy as np
from distpy import GaussianDistribution, DistributionSet,\
    DistributionHarmonizer, bivariate_histogram

fontsize = 24

# true marginal distributions are normal with mean 0 and variance 1.
correlation_coefficient = 0.5
true_mean = [0, 0]
true_covariance = [[1, correlation_coefficient], [correlation_coefficient, 1]]

marginal_distribution_set =\
    DistributionSet([(GaussianDistribution(0, 1), 'x', None)])
def conditional_solver(dictionary):
    return DistributionSet([(GaussianDistribution(correlation_coefficient *\
        dictionary['x'], 1 - (correlation_coefficient ** 2)), 'y', None)])
marginal_draws = 10000
conditional_draws = 1
ndraw = conditional_draws * marginal_draws

distribution_harmonizer = DistributionHarmonizer(marginal_distribution_set,\
    conditional_solver, marginal_draws, conditional_draws=conditional_draws)
start_time = time.time()
joint_distribution_set = distribution_harmonizer.joint_distribution_set
middle_time = time.time()
sample = joint_distribution_set.draw(ndraw)
end_time = time.time()
form_duration = middle_time - start_time
draw_duration = end_time - middle_time
print(("It took {0:.5f} s to form the joint distribution set with " +\
    "ndraw={1:d} from a DistributionHarmonizer and {2:.5f} s to return the " +\
    "sample from that distribution.").format(form_duration, ndraw,\
    draw_duration))

bivariate_histogram(sample['x'], sample['y'], reference_value_mean=true_mean,\
    reference_value_covariance=true_covariance, bins=50,\
    matplotlib_function='contourf', xlabel='$X$', ylabel='$Y$', title='',\
    fontsize=16, contour_confidence_levels=[0.68, 0.95],\
    reference_color='C3', reference_alpha=0.5, colors=['C0', 'C2'], alpha=0.5,\
    show=True)

