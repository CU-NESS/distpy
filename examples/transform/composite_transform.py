"""
File: examples/transform/composite_transform.py
Author: Keith Tauscher
Date: 11 Dec 2017

Description: Example showing how to use the composite transform.
"""
import numpy as np
from distpy import PowerTransform, AffineTransform, CompositeTransform
from test_transform import test_transform

inner_transform = AffineTransform(1, 2)
outer_transform = PowerTransform(2)
transform = CompositeTransform(inner_transform, outer_transform)
x_values = np.linspace(0.001, 1, 100)
func = (lambda x : ((x + 2) ** 2))
deriv = (lambda x : ((2 * x) + 4))
log_deriv = (lambda x : (np.log((2 * x) + 4)))
deriv2 = (lambda x : ((0 * x) + 2))
deriv_log_deriv = (lambda x : (1 / (x + 2)))
deriv3 = (lambda x : (0 * x))
deriv2_log_deriv = (lambda x : ((-1) / func(x)))
test_transform(transform, x_values, func, deriv, log_deriv, deriv2,\
    deriv_log_deriv, deriv3, deriv2_log_deriv)

