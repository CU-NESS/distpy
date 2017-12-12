"""
File: examples/transform/arcsin_transform.py
Author: Keith Tauscher
Date: 11 Dec 2017

Description: Example showing how to use the arcsin transform.
"""
import numpy as np
from distpy import ArcsinTransform
from test_transform import test_transform

transform = ArcsinTransform()
x_values = np.linspace(-0.99, 0.99, 100)
func = (lambda x : np.arcsin(x))
inv = (lambda y : np.sin(y))
deriv = (lambda x : np.power(1 - (x ** 2), -0.5))
log_deriv = (lambda x : (np.log(1 - (x ** 2)) / (-2.)))
deriv2 = (lambda x : (x * np.power(1 - (x ** 2), -1.5)))
deriv_log_deriv = (lambda x : (x / (1 - (x ** 2))))
deriv3 = (lambda x : ((1 + (2 * (x ** 2))) * np.power(1 - (x ** 2), -2.5)))
deriv2_log_deriv = (lambda x : ((1 + (x ** 2)) / ((1 - (x ** 2)) ** 2)))
test_transform(transform, x_values, func, deriv, log_deriv, deriv2,\
    deriv_log_deriv, deriv3, deriv2_log_deriv, inverse_function=inv)

