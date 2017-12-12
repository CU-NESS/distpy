"""
File: examples/transform/product_transform.py
Author: Keith Tauscher
Date: 11 Dec 2017

Description: Example showing how to use the product transform.
"""
import numpy as np
from distpy import ProductTransform, NullTransform
from test_transform import test_transform

transform = ProductTransform(NullTransform(), NullTransform())
x_values = np.linspace(1, 10, 91)
func = (lambda x : (x ** 2))
deriv = (lambda x : (2 * x))
log_deriv = (lambda x : (np.log(x) + np.log(2)))
deriv2 = (lambda x : ((0 * x) + 2))
deriv_log_deriv = (lambda x : (1 / x))
deriv3 = (lambda x : (0 * x))
deriv2_log_deriv = (lambda x : ((-1) / (x ** 2)))
test_transform(transform, x_values, func, deriv, log_deriv, deriv2,\
    deriv_log_deriv, deriv3, deriv2_log_deriv)

