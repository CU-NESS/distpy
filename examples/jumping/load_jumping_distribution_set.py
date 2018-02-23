import os
from distpy import UniformJumpingDistribution, GaussianJumpingDistribution,\
    JumpingDistributionSet

jumping_distribution_set = JumpingDistributionSet()

jumping_distribution_set.add_distribution(UniformJumpingDistribution(1), 'a')
jumping_distribution_set.add_distribution(GaussianJumpingDistribution(1), 'b')
jumping_distribution_set.add_distribution(UniformJumpingDistribution(1), 'c')

hdf5_file_name = 'TEST_DELETE_THIS.hdf5'
jumping_distribution_set.save(hdf5_file_name)
try:
    assert\
        jumping_distribution_set == JumpingDistributionSet.load(hdf5_file_name)
except:
    os.remove(hdf5_file_name)
    raise
else:
    os.remove(hdf5_file_name)

