"""
File: examples/transform/box_cox_transform.py
Author: Keith Tauscher
Date: 2 Oct 2018

Description: Example showing how to use the BoxCoxTransform class.
"""
import os
import numpy as np
from distpy import BoxCoxTransform, cast_to_transform,\
    load_transform_from_hdf5_file

num_channels = 100
x_values = np.linspace(-10, 10, num_channels)
null_transform = BoxCoxTransform(1, offset=1)

hdf5_file_name = 'TESTING_BOXCOX_TRANSFORM_CLASS.hdf5'
try:
    null_transform.save(hdf5_file_name)
    assert(null_transform == load_transform_from_hdf5_file(hdf5_file_name))
except:
    if os.path.exists(hdf5_file_name):
        os.remove(hdf5_file_name)
    raise
else:
    os.remove(hdf5_file_name)

assert(null_transform == cast_to_transform('box-cox 1 1'))

assert(np.allclose(null_transform(x_values), x_values))
assert(\
    np.allclose(null_transform.derivative(x_values), x_values ** 0))
assert(np.allclose(null_transform.second_derivative(x_values),\
    np.zeros_like(x_values)))
assert(np.allclose(null_transform.third_derivative(x_values),\
    np.zeros_like(x_values)))

