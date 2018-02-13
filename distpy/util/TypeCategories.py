"""
File: distpy/TypeCategories.py
Author: Keith Tauscher
Date: 12 Feb 2018

Description: File containing categories of types.
"""
import numpy as np

bool_types = [bool, np.bool_]
int_types = [int, np.int16, np.int32, np.int64]
float_types = [float, np.float32, np.float64, np.float128]
real_numerical_types = int_types + float_types
complex_numerical_types = [complex, np.complex64, np.complex128, np.complex256]
numerical_types = real_numerical_types + complex_numerical_types
sequence_types = [list, tuple, np.ndarray, np.matrix]
