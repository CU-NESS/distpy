"""
File: examples/transform/test_transform.py
Author: Keith Tauscher
Date: 11 Dec 2017

Description: File containing function used to test each Transform class in the
             other scripts in examples/transform directory.
"""
import os
import numpy as np
from distpy import load_transform_from_hdf5_file

def test_transform(transform, x_values, function, derivative, log_derivative,\
    second_derivative, derivative_of_log_derivative, third_derivative,\
    second_derivative_of_log_derivative, inverse_function=None, **tol_kwargs):
    """
    Tests the given transforms, its derivatives, and (if it exists) its inverse.
    
    transform: a Transform object
    x_values: 1D numpy.ndarray of valid x values for this transform
    function, derivative, log_derivative,
    second_derivative, derivative_of_log_derivative,
    third_derivative, second_derivative_of_log_derivative: function objects
                                                           which accept 1D
                                                           numpy.ndarray
                                                           objects and match
                                                           expected behavior of
                                                           given transform
    inverse_function: if transform has inverse, it should be supplied as a
                      function object which accepts 1D numpy.ndarray objects
    
    raises: AssertionError if transform does not match expected behavior
    """
    y_values = function(x_values)
    conditions =\
    [\
        np.allclose(transform(x_values), y_values, **tol_kwargs),\
        np.allclose(transform.derivative(x_values), derivative(x_values),\
            **tol_kwargs),\
        np.allclose(transform.log_derivative(x_values),\
            log_derivative(x_values), **tol_kwargs),\
        np.allclose(transform.second_derivative(x_values),\
            second_derivative(x_values), **tol_kwargs),\
        np.allclose(transform.derivative_of_log_derivative(x_values),\
            derivative_of_log_derivative(x_values), **tol_kwargs),\
        np.allclose(transform.third_derivative(x_values),\
            third_derivative(x_values), **tol_kwargs),\
        np.allclose(transform.second_derivative_of_log_derivative(x_values),\
            second_derivative_of_log_derivative(x_values), **tol_kwargs)\
    ]
    if inverse_function is not None:
        conditions.append(np.allclose(transform.apply_inverse(y_values),\
            inverse_function(y_values), **tol_kwargs))
    if not all(conditions):
        raise AssertionError("At least one condition was not true.")
    file_name = 'transform_TEST.hdf5'
    transform.save(file_name)
    try:
        transform == load_transform_from_hdf5_file(file_name)
    except:
        os.remove(file_name)
        raise
    else:
        os.remove(file_name)

