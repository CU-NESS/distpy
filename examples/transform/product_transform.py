"""
File: examples/transform/product_transform.py
Author: Keith Tauscher
Date: 6 Dec 2017

Description: Example showing how to use the product transform.
"""
import os
import numpy as np
from distpy import SquareTransform, NullTransform, ProductTransform,\
    load_transform_from_hdf5_file

transforms = [NullTransform(), SquareTransform()]
transform = ProductTransform(*transforms)
conditions =\
[\
    transform(2) == 8,\
    transform.log_derivative(2) == np.log(12)\
]


if not all(conditions):
    raise AssertionError("ProductTransform test failed at least one condition.")

file_name = 'product_transform_TEST.hdf5'
transform.save(file_name)
try:
    assert transform == load_transform_from_hdf5_file(file_name)
except:
    os.remove(file_name)
    raise
else:
    os.remove(file_name)


