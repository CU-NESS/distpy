"""
File: examples/transform/square_transform.py
Author: Keith Tauscher
Date: 10 Sep 2017

Description: Example showing how to use the square transform.
"""
import os
import numpy as np
from distpy import SquareTransform, load_transform_from_hdf5_file

transform = SquareTransform()
conditions =\
[\
    np.isclose(transform(1), 1),\
    np.isclose(transform.I(1), 1),\
    np.isclose(transform.log_value_addition(1), np.log(2))\
]
if not all(conditions):
    raise AssertionError("SquareTransform test failed at least one condition.")

file_name = 'square_transform_TEST.hdf5'
transform.save(file_name)
try:
    assert transform == load_transform_from_hdf5_file(file_name)
except:
    os.remove(file_name)
    raise
else:
    os.remove(file_name)

