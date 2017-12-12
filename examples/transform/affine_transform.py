"""
File: examples/transform/affine_transform.py
Author: Keith Tauscher
Date: 11 Dec 2017

Description: Example showing how to use the affine transform.
"""
import numpy as np
from distpy import AffineTransform
from test_transform import test_transform

transform = AffineTransform(2, 3)
x_values = np.linspace(-1, 1, 100)
func = (lambda x : ((2 * x) + 3))
inv = (lambda y : ((y - 3) / 2.))
deriv = (lambda x : ((0 * x) + 2))
log_deriv = (lambda x : ((0 * x) + np.log(2)))
deriv2 = (lambda x : (0 * x))
deriv_log_deriv = (lambda x : (0 * x))
deriv3 = (lambda x : (0 * x))
deriv2_log_deriv = (lambda x : (0 * x))
test_transform(transform, x_values, func, deriv, log_deriv, deriv2,\
    deriv_log_deriv, deriv3, deriv2_log_deriv, inverse_function=inv)

