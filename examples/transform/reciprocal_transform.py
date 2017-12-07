"""
File: examples/transform/reciprocal_transform.py
Author: Keith Tauscher
Date: 6 Dec 2017

Description: Example showing how to use the reciprocal transform.
"""
import os
import numpy as np
from distpy import NullTransform, ReciprocalTransform,\
    load_transform_from_hdf5_file

transform = ReciprocalTransform(NullTransform())
conditions =\
[\
    np.isclose(transform(3), 0.333333333, rtol=0, atol=1e-6),\
    np.isclose(transform.I(0.333333333), 3., rtol=0, atol=1e-6),\
    np.isclose(transform.log_derivative(1), 0, rtol=0, atol=1e-6)\
]


if not all(conditions):
    raise AssertionError("ReciprocalTransform test failed at least one " +\
        "condition.")

file_name = 'reciprocal_transform_TEST.hdf5'
transform.save(file_name)
try:
    assert transform == load_transform_from_hdf5_file(file_name)
except:
    os.remove(file_name)
    raise
else:
    os.remove(file_name)


