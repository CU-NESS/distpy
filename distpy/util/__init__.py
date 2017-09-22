"""
File: distpy/util/__init__.py
Author: Keith Tauscher
Date: 10 Sep 2017

Description: Imports classes in this module so that any given class, CLASS, can
             be imported using "from distpy import CLASS"
"""
from distpy.util.Savable import Savable
from distpy.util.TypeCategories import bool_types, int_types, float_types,\
    numerical_types, sequence_types

