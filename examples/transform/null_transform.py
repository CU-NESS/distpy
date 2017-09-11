"""
File: examples/transform/null_transform.py
Author: Keith Tauscher
Date: 10 Sep 2017

Description: Example showing how to use the null transform.
"""
import os
from distpy import NullTransform, load_transform_from_hdf5_file

transform = NullTransform()
conditions =\
[\
    transform(1) == 1,\
    transform.I(1) == 1,\
    transform.log_value_addition(1) == 0\
]
if not all(conditions):
    raise AssertionError("NullTransform test failed at least one condition.")

file_name = 'null_transform_TEST.hdf5'
transform.save(file_name)
try:
    assert transform == load_transform_from_hdf5_file(file_name)
except:
    os.remove(file_name)
    raise
else:
    os.remove(file_name)


