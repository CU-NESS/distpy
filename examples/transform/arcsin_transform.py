"""
File: examples/transform/arcsin_transform.py
Author: Keith Tauscher
Date: 10 Sep 2017

Description: Example showing how to use the arcsin transform.
"""
import os
import numpy as np
from distpy import ArcsinTransform, load_transform_from_hdf5_file

transform = ArcsinTransform()
conditions =\
[\
    np.isclose(transform(0), 0),\
    np.isclose(transform.I(0), 0),\
    np.isclose(transform.log_value_addition(0), 0)\
]
if not all(conditions):
    raise AssertionError("ArcsinTransform test failed at least one condition.")

file_name = 'Arcsin_transform_TEST.hdf5'
transform.save(file_name)
try:
    assert transform == load_transform_from_hdf5_file(file_name)
except:
    os.remove(file_name)
    raise
else:
    os.remove(file_name)

