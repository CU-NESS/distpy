"""
File: examples/transform/sum_transform.py
Author: Keith Tauscher
Date: 11 Dec 2017

Description: Example showing how to use the sum transform.
"""
import numpy as np
from distpy import SumTransform, NullTransform
from test_transform import test_transform

transform = SumTransform(NullTransform(), NullTransform())
x_values = np.linspace(-1, 1, 10)
func = (lambda x : (2 * x))
deriv = (lambda x : ((0 * x) + 2))
log_deriv = (lambda x : (np.log(2)))
deriv2 = (lambda x : (0 * x))
deriv_log_deriv = (lambda x : (0 * x))
deriv3 = (lambda x : (0 * x))
deriv2_log_deriv = (lambda x : (0 * x))
test_transform(transform, x_values, func, deriv, log_deriv, deriv2,\
    deriv_log_deriv, deriv3, deriv2_log_deriv)
