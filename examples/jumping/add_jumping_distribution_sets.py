"""
File: examples/jumping/add_jumping_distribution_sets.py
Author: Keith Tauscher
Date: 18 Feb 2018

Description: Example script showing the meaning and method of summing two
             JumpingDistributionSet objects using the '+' and '+=' operators.
"""
from distpy import UniformJumpingDistribution, JumpingDistributionSet

jumping_distribution_set = JumpingDistributionSet()
jumping_distribution_set.add_distribution(UniformJumpingDistribution(1), 'a')
jumping_distribution_set.add_distribution(UniformJumpingDistribution(2), 'b')
jumping_distribution_set_2 = JumpingDistributionSet()
jumping_distribution_set_2.add_distribution(UniformJumpingDistribution(3), 'c')
combined_jumping_distribution_set =\
    jumping_distribution_set + jumping_distribution_set_2
assert(jumping_distribution_set.params == ['a', 'b'])
assert(jumping_distribution_set_2.params == ['c'])
assert(combined_jumping_distribution_set.params == ['a', 'b', 'c'])
assert(combined_jumping_distribution_set != jumping_distribution_set)
jumping_distribution_set += jumping_distribution_set_2
assert(combined_jumping_distribution_set == jumping_distribution_set)
