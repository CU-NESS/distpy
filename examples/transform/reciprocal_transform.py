"""
File: examples/transform/reciprocal_transform.py
Author: Keith Tauscher
Date: 11 Dec 2017

Description: Example showing how to use the reciprocal transform.
"""
from __future__ import division
import numpy as np
from distpy import ReciprocalTransform
from test_transform import test_transform

transform = ReciprocalTransform()
x_values = np.linspace(1, 10, 91)
func = (lambda x : (1 / x))
inv = (lambda x : (1 / x))
deriv = (lambda x : ((-1) / (x ** 2)))
log_deriv = (lambda x : ((-2) * np.log(x)))
deriv2 = (lambda x : (2 / (x ** 3)))
deriv_log_deriv = (lambda x : ((-2) / x))
deriv3 = (lambda x : ((-6) / (x ** 4)))
deriv2_log_deriv = (lambda x : (2 / (x ** 2)))
test_transform(transform, x_values, func, deriv, log_deriv, deriv2,\
    deriv_log_deriv, deriv3, deriv2_log_deriv, inverse_function=inv)

