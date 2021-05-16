"""
Module containing categories of types to use for type-checking inputs. For
example, functions that use numpy arrays under the hood can check if an input
variable `value` is a sequence using `type(value) in sequence_types` before
casting to an array.

**File**: $DISTPY/distpy/util/TypeCategories.py  
**Author**: Keith Tauscher  
**Date**: 15 May 2021
"""
import numpy as np

bool_types = [bool, np.bool_]
"""Types representing booleans."""

int_types = [int, np.int16, np.int32, np.int64]
"""Types representing integers."""

float_types = [float, np.float32, np.float64, np.float128]
"""Types representing floats."""

real_numerical_types = int_types + float_types
"""Types representing real numbers, including integers and floats."""

complex_numerical_types = [complex, np.complex64, np.complex128, np.complex256]
"""Types representing complex numbers."""

numerical_types = real_numerical_types + complex_numerical_types
"""Types representing numbers, including real numbers and complex numbers."""

sequence_types = [list, tuple, np.ndarray, np.matrix]
"""Types representing sequences like lists, tuples, arrays, and matrices."""

