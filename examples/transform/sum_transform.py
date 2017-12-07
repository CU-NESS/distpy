"""
File: examples/transform/sum_transform.py
Author: Keith Tauscher
Date: 6 Dec 2017

Description: Example showing how to use the sum transform.
"""
import os
import numpy as np
from distpy import SquareTransform, NullTransform, SumTransform,\
    load_transform_from_hdf5_file

transforms = [NullTransform(), SquareTransform()]
transform = SumTransform(*transforms)
conditions =\
[\
    transform(1) == 2,\
    transform.log_derivative(1) == np.log(3)\
]


if not all(conditions):
    raise AssertionError("SumTransform test failed at least one condition.")

file_name = 'sum_transform_TEST.hdf5'
transform.save(file_name)
try:
    assert transform == load_transform_from_hdf5_file(file_name)
except:
    os.remove(file_name)
    raise
else:
    os.remove(file_name)


