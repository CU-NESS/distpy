"""
File: examples/transform/exponential_transform.py
Author: Keith Tauscher
Date: 6 Dec 2017

Description: Example showing how to use the exponential transform.
"""
import os
import numpy as np
from distpy import ExponentialTransform, cast_to_transform,\
    load_transform_from_hdf5_file

transform = ExponentialTransform()
conditions =\
[\
    transform == cast_to_transform('exp'),\
    np.allclose(transform(np.array([1, np.log(2)])), np.array([np.e, 2])),\
    np.isclose(transform.I(np.e), 1),\
    np.isclose(transform.log_derivative(1), 1),\
    np.isclose(transform.derivative_of_log_derivative(1), 1),\
    np.isclose(transform.second_derivative_of_log_derivative(1), 0)\
]
if not all(conditions):
    raise AssertionError("ExponentialTransform test failed at least one " +\
        "condition.")

file_name = 'exponential_transform_TEST.hdf5'
transform.save(file_name)
try:
    assert transform == load_transform_from_hdf5_file(file_name)
except:
    os.remove(file_name)
    raise
else:
    os.remove(file_name)


