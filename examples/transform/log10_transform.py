"""
File: examples/transform/log10_transform.py
Author: Keith Tauscher
Date: 10 Sep 2017

Description: Example showing how to use the log transform.
"""
import os
import numpy as np
from distpy import Log10Transform, load_transform_from_hdf5_file

transform = Log10Transform()
conditions =\
[\
    np.isclose(transform(1), 0),\
    np.isclose(transform.I(1), 10),\
    np.isclose(transform.log_derivative(1), -np.log(np.log(10)))\
]
if not all(conditions):
    raise AssertionError("Log10Transform test failed at least one condition.")

file_name = 'log10_transform_TEST.hdf5'
transform.save(file_name)
try:
    assert transform == load_transform_from_hdf5_file(file_name)
except:
    os.remove(file_name)
    raise
else:
    os.remove(file_name)


