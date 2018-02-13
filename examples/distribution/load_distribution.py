import os
from distpy import UniformDistribution, load_distribution_from_hdf5_file

distribution = UniformDistribution(-1, 6, metadata='a')
hdf5_file_name = 'TEST_DELETE_THIS.hdf5'
distribution.save(hdf5_file_name)
try:
    assert distribution == load_distribution_from_hdf5_file(hdf5_file_name)
except:
    os.remove(hdf5_file_name)
    raise
else:
    os.remove(hdf5_file_name)

