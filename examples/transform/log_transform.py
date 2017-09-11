"""
File: examples/transform/log_transform.py
Author: Keith Tauscher
Date: 10 Sep 2017

Description: Example showing how to use the log transform.
"""
import os
import numpy as np
from distpy import LogTransform, load_transform_from_hdf5_file

transform = LogTransform()
conditions =\
[\
    np.isclose(transform(1), 0),\
    np.isclose(transform.I(1), np.e),\
    np.isclose(transform.log_value_addition(1), 0)\
]
if not all(conditions):
    raise AssertionError("LogTransform test failed at least one condition.")

file_name = 'log_transform_TEST.hdf5'
transform.save(file_name)
try:
    assert transform == load_transform_from_hdf5_file(file_name)
except:
    os.remove(file_name)
    raise
else:
    os.remove(file_name)


