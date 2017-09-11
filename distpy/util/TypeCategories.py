"""
File: distpy/TypeCategories.py
Author: Keith Tauscher
Date: 6 Aug 2017

Description: File containing categories of types.
"""
import numpy as np

int_types = [int, np.int16, np.int32, np.int64]
float_types = [float, np.float32, np.float64]
numerical_types = int_types + float_types
sequence_types = [list, tuple, np.ndarray, np.matrix]
