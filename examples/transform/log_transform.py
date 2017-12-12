"""
File: examples/transform/log_transform.py
Author: Keith Tauscher
Date: 11 Dec 2017

Description: Example showing how to use the log transform.
"""
import numpy as np
from distpy import LogTransform
from test_transform import test_transform

transform = LogTransform()
x_values = np.linspace(1, 10, 91)
func = (lambda x : (np.log(x)))
inv = (lambda y : (np.exp(y)))
deriv = (lambda x : (1 / x))
log_deriv = (lambda x : (-np.log(x)))
deriv2 = (lambda x : ((-1) / (x ** 2)))
deriv_log_deriv = (lambda x : ((-1) / x))
deriv3 = (lambda x : (2 / (x ** 3)))
deriv2_log_deriv = (lambda x : (1 / (x ** 2)))
test_transform(transform, x_values, func, deriv, log_deriv, deriv2,\
    deriv_log_deriv, deriv3, deriv2_log_deriv, inverse_function=inv)


