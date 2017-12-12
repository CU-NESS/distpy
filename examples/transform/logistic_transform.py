"""
File: examples/transform/logistic_transform.py
Author: Keith Tauscher
Date: 11 Dec 2017

Description: Example showing how to use the logistic transform.
"""
import numpy as np
from distpy import LogisticTransform
from test_transform import test_transform

transform = LogisticTransform()
x_values = np.linspace(0.01, 0.99, 99)
func = (lambda x : (np.log(x / (1 - x))))
inv = (lambda y : (1 / (1 + np.exp(-y))))
deriv = (lambda x : (1 / (x * (1 - x))))
log_deriv = (lambda x : (-(np.log(x) + np.log(1 - x))))
deriv2 = (lambda x : ((1 / ((1 - x) ** 2)) - (1 / (x ** 2))))
deriv_log_deriv = (lambda x : ((1 / (1 - x)) - (1 / x)))
deriv3 = (lambda x : (2 * ((1 / (x ** 3)) + (1 / ((1 - x) ** 3))) ))
deriv2_log_deriv = (lambda x : ((1 / (x ** 2)) + (1 / ((1 - x) ** 2))))
test_transform(transform, x_values, func, deriv, log_deriv, deriv2,\
    deriv_log_deriv, deriv3, deriv2_log_deriv, inverse_function=inv)


