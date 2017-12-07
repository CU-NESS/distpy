"""
File: examples/transform/reciprocal_transform.py
Author: Keith Tauscher
Date: 6 Dec 2017

Description: Example showing how to use the reciprocal transform.
"""
import os
import numpy as np
from distpy import ExponentiatedTransform, LogTransform,\
    load_transform_from_hdf5_file

transform = ExponentiatedTransform(LogTransform())
tol_kwargs = {'rtol': 0, 'atol': 1e-6}
conditions =\
[\
    np.isclose(transform(291), 291, **tol_kwargs),\
    np.isclose(transform.I(1/3.), 1/3., **tol_kwargs),\
    np.isclose(transform.derivative(np.pi), 1, **tol_kwargs),\
    np.isclose(transform.second_derivative(np.e), 0, **tol_kwargs)\
]

if not all(conditions):
    raise AssertionError("ExponentiatedTransform test failed at least one " +\
        "condition.")

file_name = 'exponentiated_transform_TEST.hdf5'
transform.save(file_name)
try:
    assert transform == load_transform_from_hdf5_file(file_name)
except:
    os.remove(file_name)
    raise
else:
    os.remove(file_name)


