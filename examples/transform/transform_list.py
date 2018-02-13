"""
File: examples/transform/transform_list.py
Author: Keith Tauscher
Date: 12 Feb 2018

Description: Example showing how to use, save, and load TransformList objects.
"""
import os
from distpy import TransformList, NullTransform, Log10Transform,\
    ArcsinTransform

transform_list = TransformList('log10', None, 'arcsin')
hdf5_file_name = 'TEST_DELETE_THIS.hdf5'
transform_list.save(hdf5_file_name)
try:
    assert transform_list == TransformList.load(hdf5_file_name)
except:
    os.remove(hdf5_file_name)
    raise
else:
    os.remove(hdf5_file_name)

assert transform_list[0] == Log10Transform()
assert transform_list[1] == NullTransform()
assert transform_list[2] == ArcsinTransform()
