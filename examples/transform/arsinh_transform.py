"""
File: examples/transform/arsinh_transform.py
Author: Keith Tauscher
Date: 2 Oct 2018

Description: Example showing how to use the ArsinhTransform class.
"""
import os
import numpy as np
from distpy import ArsinhTransform, cast_to_transform,\
    load_transform_from_hdf5_file

num_channels = 100
x_values = np.linspace(-10, 10, num_channels)
sinh_transform = ArsinhTransform(1)

hdf5_file_name = 'TESTING_ARSINH_TRANSFORM_CLASS.hdf5'
try:
    sinh_transform.save(hdf5_file_name)
    assert(sinh_transform == load_transform_from_hdf5_file(hdf5_file_name))
except:
    if os.path.exists(hdf5_file_name):
        os.remove(hdf5_file_name)
    raise
else:
    os.remove(hdf5_file_name)

assert(sinh_transform == cast_to_transform('arsinh 1'))
assert(sinh_transform == cast_to_transform('arsinh 1 0'))

assert(np.allclose(sinh_transform(x_values), np.sinh(x_values)))
assert(np.allclose(sinh_transform.derivative(x_values), np.cosh(x_values)))
assert(\
    np.allclose(sinh_transform.second_derivative(x_values), np.sinh(x_values)))
assert(\
    np.allclose(sinh_transform.third_derivative(x_values), np.cosh(x_values)))

