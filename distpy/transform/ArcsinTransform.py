"""
File: distpy/transform/ArcsinTransform.py
Author: Keith Tauscher
Date: 12 Feb 2018

Description: File containing class representing transform which takes arcsin of
             inputs.
"""
import numpy as np
from .Transform import Transform

class ArcsinTransform(Transform):
    """
    Class representing a transform based on the inverse sine function.
    """
    def derivative(self, value):
        """
        Computes the derivative of the function underlying this Transform at
        the given value(s).
        
        value: single number or numpy.ndarray of values
        
        returns: value of derivative in same format as value
        """
        return np.power(1. - (value ** 2), -0.5)
    
    def second_derivative(self, value):
        """
        Computes the second derivative of the function underlying this
        Transform at the given value(s).
        
        value: single number or numpy.ndarray of values
        
        returns: value of second derivative in same format as value
        """
        return value * np.power(1 - (value ** 2), -1.5)
    
    def third_derivative(self, value):
        """
        Computes the third derivative of the function underlying this Transform
        at the given value(s).
        
        value: single number or numpy.ndarray of values
        
        returns: value of third derivative in same format as value
        """
        value2 = (value ** 2)
        return ((2. * value2) + 1.) / np.power(1. - (value2), 2.5)
    
    def log_derivative(self, value):
        """
        Computes the natural logarithm of the derivative of the function
        underlying this Transform at the given value(s).
        
        value: single number or numpy.ndarray of values
        
        returns: value of log derivative in same format as value
        """
        return -0.5 * np.log(1. - (value ** 2))
    
    def derivative_of_log_derivative(self, value):
        """
        Computes the derivative of the natural logarithm of the derivative of
        the function underlying this Transform at the given value(s).
        
        value: single number or numpy.ndarray of values
        
        returns: value of derivative of log derivative in same format as value
        """
        return (value / (1. - (value ** 2)))
    
    def second_derivative_of_log_derivative(self, value):
        """
        Computes the second derivative of the natural logarithm of the
        derivative of the function underlying this Transform at the given
        value(s).
        
        value: single number or numpy.ndarray of values
        
        returns: value of second derivative of log derivative in same format as
                 value
        """
        value2 = (value ** 2)
        return ((1. + value2) / ((1. - value2) ** 2))
    
    def apply(self, value):
        """
        Applies this transform to the value and returns the result.
        
        value: single number or numpy.ndarray of values
        
        returns: value of function in same format as value
        """
        return np.arcsin(value)
    
    def apply_inverse(self, value):
        """
        Applies the inverse of this transform to the value.
        
        value: single number or numpy.ndarray of values
        
        returns: value of inverse function in same format as value
        """
        return np.sin(value)
    
    def __eq__(self, other):
        """
        Checks for equality with other. Returns True iff other is a
        ArcsinTransform.
        """
        return isinstance(other, ArcsinTransform)
    
    def to_string(self):
        """
        Generates a string version of this Transform.
        
        returns: value which can be cast into this Transform
        """
        return 'arcsin'
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data about this transform.
        
        group: hdf5 file group to which to write data about this transform
        """
        group.attrs['class'] = 'ArcsinTransform'

