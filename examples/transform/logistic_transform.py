"""
File: examples/transform/logistic_transform.py
Author: Keith Tauscher
Date: 10 Sep 2017

Description: Example showing how to use the logistic transform.
"""
import os
import numpy as np
from distpy import LogisticTransform, load_transform_from_hdf5_file

transform = LogisticTransform()
conditions =\
[\
    np.isclose(transform(0.5), 0),\
    np.isclose(transform.I(0), 0.5),\
    np.isclose(transform.log_value_addition(0.5), 2 * np.log(2))\
]
if not all(conditions):
    raise AssertionError("LogisticTransform test failed at least one " +\
                         "condition.")

file_name = 'Logistic_transform_TEST.hdf5'
transform.save(file_name)
try:
    assert transform == load_transform_from_hdf5_file(file_name)
except:
    os.remove(file_name)
    raise
else:
    os.remove(file_name)

