"""
File: examples/transform/null_transform.py
Author: Keith Tauscher
Date: 11 Dec 2017

Description: Example showing how to use the null transform.
"""
import numpy as np
from distpy import NullTransform
from test_transform import test_transform

transform = NullTransform()
x_values = np.linspace(-1, 1, 100)
func = (lambda x : x)
inv = (lambda y : y)
deriv = (lambda x : ((0 * x) + 1))
log_deriv = (lambda x : (0 * x))
deriv2 = (lambda x : (0 * x))
deriv_log_deriv = (lambda x : (0 * x))
deriv3 = (lambda x : (0 * x))
deriv2_log_deriv = (lambda x : (0 * x))
test_transform(transform, x_values, func, deriv, log_deriv, deriv2,\
    deriv_log_deriv, deriv3, deriv2_log_deriv, inverse_function=inv)


