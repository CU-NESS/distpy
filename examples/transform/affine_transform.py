"""
File: examples/transform/affine_transform.py
Author: Keith Tauscher
Date: 3 Dec 2017

Description: Example showing how to use the affine transform.
"""
import os
import numpy as np
from distpy import AffineTransform, load_transform_from_hdf5_file

transform = AffineTransform(2, 3)
conditions =\
[\
    transform(1) == 5,\
    transform.I(5) == 1,\
    transform.log_derivative(1) == np.log(2)\
]
if not all(conditions):
    raise AssertionError("AffineTransform test failed at least one condition.")

file_name = 'affine_transform_TEST.hdf5'
transform.save(file_name)
try:
    assert transform == load_transform_from_hdf5_file(file_name)
except:
    os.remove(file_name)
    raise
else:
    os.remove(file_name)


