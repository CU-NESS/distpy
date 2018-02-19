from distpy import UniformDistribution, DistributionSet

distribution_set = DistributionSet()
distribution_set.add_distribution(UniformDistribution(), 'a')
distribution_set.add_distribution(UniformDistribution(), 'b')
distribution_set_2 = DistributionSet()
distribution_set_2.add_distribution(UniformDistribution(), 'c')
combined_distribution_set = distribution_set + distribution_set_2
assert(distribution_set.params == ['a', 'b'])
assert(distribution_set_2.params == ['c'])
assert(combined_distribution_set.params == ['a', 'b', 'c'])
assert(combined_distribution_set != distribution_set)
distribution_set += distribution_set_2
assert(combined_distribution_set == distribution_set)
