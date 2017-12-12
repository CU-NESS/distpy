"""
File: examples/transform/exponential_transform.py
Author: Keith Tauscher
Date: 11 Dec 2017

Description: Example showing how to use the exponential transform.
"""
import numpy as np
from distpy import ExponentialTransform
from test_transform import test_transform

transform = ExponentialTransform()
x_values = np.linspace(-1, 1, 100)
func = (lambda x : (np.exp(x)))
inv = (lambda y : (np.log(y)))
deriv = (lambda x : (func(x)))
log_deriv = (lambda x : (x))
deriv2 = (lambda x : (func(x)))
deriv_log_deriv = (lambda x : ((0 * x) + 1))
deriv3 = (lambda x : (func(x)))
deriv2_log_deriv = (lambda x : (0 * x))
test_transform(transform, x_values, func, deriv, log_deriv, deriv2,\
    deriv_log_deriv, deriv3, deriv2_log_deriv, inverse_function=inv)


