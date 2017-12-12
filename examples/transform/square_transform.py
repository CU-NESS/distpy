"""
File: examples/transform/square_transform.py
Author: Keith Tauscher
Date: 11 Dec 2017

Description: Example showing how to use the square transform.
"""
import numpy as np
from distpy import SquareTransform
from test_transform import test_transform

transform = SquareTransform()
x_values = np.linspace(1, 10, 91)
func = (lambda x : (x ** 2))
inv = (lambda y : (np.sqrt(y)))
deriv = (lambda x : (2 * x))
log_deriv = (lambda x : (np.log(2 * x)))
deriv2 = (lambda x : ((0 * x) + 2))
deriv_log_deriv = (lambda x : (1 / x))
deriv3 = (lambda x : (0 * x))
deriv2_log_deriv = (lambda x : ((-1) / (x ** 2)))
test_transform(transform, x_values, func, deriv, log_deriv, deriv2,\
    deriv_log_deriv, deriv3, deriv2_log_deriv, inverse_function=inv)


