"""
File: distpy/transform/LoggedTransform.py
Author: Keith Tauscher
Date: 12 Feb 2017

Description: File containing class describing the natural logarithm of an
             arbitrary transformation.
"""
import numpy as np
from .Transform import Transform

class LoggedTransform(Transform):
    """
    Class representing the logarithm of an arbitrary transformation.
    """
    def __init__(self, transform):
        """
        Initializes a new LoggedTransform of the given transform.
        
        transform: must be a Transform object
        """
        self.transform = transform
    
    @property
    def transform(self):
        """
        Property storing the transform which this is the logarithm of.
        """
        if not hasattr(self, '_transform'):
            raise AttributeError("transform referenced before it was set.")
        return self._transform
    
    @transform.setter
    def transform(self, value):
        """
        Setter for the transform which this is the logarithm of.
        
        value: must be a Transform object
        """
        if isinstance(value, Transform):
            self._transform = value
        else:
            raise TypeError("transform was not a Transform object.")
    
    def derivative(self, value):
        """
        Computes the derivative of the function underlying this Transform at
        the given value(s).
        
        value: single number or numpy.ndarray of values
        
        returns: value of derivative in same format as value
        """
        return self.transform.derivative(value) / self.transform(value)
    
    def second_derivative(self, value):
        """
        Computes the second derivative of the function underlying this
        Transform at the given value(s).
        
        value: single number or numpy.ndarray of values
        
        returns: value of second derivative in same format as value
        """
        func = self.transform(value)
        func_deriv = self.transform.derivative(value)
        func_deriv2 = self.transform.second_derivative(value)
        return ((func_deriv2 * func) - (func_deriv ** 2)) / (func ** 2)
    
    def third_derivative(self, value):
        """
        Computes the third derivative of the function underlying this Transform
        at the given value(s).
        
        value: single number or numpy.ndarray of values
        
        returns: value of third derivative in same format as value
        """
        func = self.transform(value)
        func_deriv = self.transform.derivative(value)
        func_deriv2 = self.transform.second_derivative(value)
        func_deriv3 = self.transform.third_derivative(value)
        return ((func_deriv3 * (func ** 2)) -\
            (3 * (func * func_deriv * func_deriv2)) +\
            (2 * (func_deriv ** 3))) / (func ** 3)
    
    def log_derivative(self, value):
        """
        Computes the natural logarithm of the derivative of the function
        underlying this Transform at the given value(s).
        
        value: single number or numpy.ndarray of values
        
        returns: value of log derivative in same format as value
        """
        return self.transform.log_derivative(value) - self.apply(value)
    
    def derivative_of_log_derivative(self, value):
        """
        Computes the derivative of the natural logarithm of the derivative of
        the function underlying this Transform at the given value(s).
        
        value: single number or numpy.ndarray of values
        
        returns: value of derivative of log derivative in same format as value
        """
        return self.transform.derivative_of_log_derivative(value) -\
            self.derivative(value)
    
    def second_derivative_of_log_derivative(self, value):
        """
        Computes the second derivative of the natural logarithm of the
        derivative of the function underlying this Transform at the given
        value(s).
        
        value: single number or numpy.ndarray of values
        
        returns: value of second derivative of log derivative in same format as
                 value
        """
        return self.transform.second_derivative_of_log_derivative(value) -\
            self.second_derivative(value)
    
    def apply(self, value):
        """
        Applies this transform to the value and returns the result.
        
        value: single number or numpy.ndarray of values
        
        returns: value of function in same format as value
        """
        return np.log(self.transform(value))
    
    def apply_inverse(self, value):
        """
        Applies the inverse of this transform to the value.
        
        value: single number or numpy.ndarray of values
        
        returns: value of inverse function in same format as value
        """
        return self.transform.apply_inverse(np.exp(value))
    
    def to_string(self):
        """
        Generates a string version of this Transform.
        
        returns: value which can be cast into this Transform
        """
        return 'ln[{!s}]'.format(self.transform.to_string())
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data about this transform.
        
        group: hdf5 file group to which to write data about this transform
        """
        group.attrs['class'] = 'LoggedTransform'
        self.transform.fill_hdf5_group(group.create_group('transform'))
    
    def __eq__(self, other):
        """
        Fills the given hdf5 file group with data about this transform.
        
        other: object to check for equality
        
        returns True if both Transforms are the same
        """
        if isinstance(other, LoggedTransform):
            return self.transform == other.transform
        else:
            return False

