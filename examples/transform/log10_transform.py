"""
File: examples/transform/log10_transform.py
Author: Keith Tauscher
Date: 11 Dec 2017

Description: Example showing how to use the log transform.
"""
import numpy as np
from distpy import Log10Transform
from test_transform import test_transform

transform = Log10Transform()
x_values = np.linspace(1, 10, 91)
func = (lambda x : (np.log10(x)))
inv = (lambda y : (np.power(10, y)))
deriv = (lambda x : (1 / (x * np.log(10))))
log_deriv = (lambda x : (-(np.log(np.log(10)) + np.log(x))))
deriv2 = (lambda x : ((-1) / ((x ** 2) * np.log(10))))
deriv_log_deriv = (lambda x : ((-1) / x))
deriv3 = (lambda x : (2 / ((x ** 3) * np.log(10))))
deriv2_log_deriv = (lambda x : (1 / (x ** 2)))
test_transform(transform, x_values, func, deriv, log_deriv, deriv2,\
    deriv_log_deriv, deriv3, deriv2_log_deriv, inverse_function=inv)

