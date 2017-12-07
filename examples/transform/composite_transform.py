"""
File: examples/transform/composite_transform.py
Author: Keith Tauscher
Date: 3 Dec 2017

Description: Example showing how to use the composite transform.
"""
import os
import numpy as np
from distpy import SquareTransform, AffineTransform, CompositeTransform,\
    load_transform_from_hdf5_file

inner_transform = AffineTransform(1, 2)
outer_transform = SquareTransform()
transform = CompositeTransform(inner_transform, outer_transform)
conditions =\
[\
    transform(1) == 9,\
    transform.I(16) == 2,\
    transform.log_derivative(1) == np.log(6)\
]


if not all(conditions):
    raise AssertionError("CompositeTransform test failed at least one " +\
        "condition.")

file_name = 'composite_transform_TEST.hdf5'
transform.save(file_name)
try:
    assert transform == load_transform_from_hdf5_file(file_name)
except:
    os.remove(file_name)
    raise
else:
    os.remove(file_name)


