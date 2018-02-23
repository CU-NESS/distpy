"""
File: examples/transform/transform_set.py
Author: Keith Tauscher
Date: 22 Feb 2018

Description: Example showing how to use, save, and load TransformSet objects.
"""
import os
from distpy import TransformSet, NullTransform, Log10Transform,\
    ArcsinTransform

transform_set = TransformSet(['log10', None, 'arcsin'], ['a', 'b', 'c'])
transform_set2 = TransformSet({'a': 'log10', 'b': None, 'c': 'arcsin'})
assert transform_set == transform_set2
hdf5_file_name = 'TEST_DELETE_THIS.hdf5'
transform_set.save(hdf5_file_name)
try:
    assert transform_set == TransformSet.load(hdf5_file_name)
except:
    os.remove(hdf5_file_name)
    raise
else:
    os.remove(hdf5_file_name)

assert transform_set['a'] == Log10Transform()
assert transform_set['b'] == NullTransform()
assert transform_set['c'] == ArcsinTransform()
