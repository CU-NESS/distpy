"""
File: distpy/transform/PowerTransform.py
Author: Keith Tauscher
Date: 3 Apr 2019

Description: File containing class representing transforms which take a power
             of the their inputs.
"""
from __future__ import division
import numpy as np
from ..util import real_numerical_types
from .Transform import Transform

class PowerTransform(Transform):
    """
    Class representing a transform based on the power function.
    """
    def __init__(self, power):
        """
        Initializes a PowerTransform.
        
        power: positive number to which power inputs are put
        """
        self.power = power
    
    @property
    def power(self):
        """
        Property storing the power at the heart of this distribution.
        """
        if not hasattr(self, '_power'):
            raise AttributeError("power was referenced before it was set.")
        return self._power
    
    @power.setter
    def power(self, value):
        """
        Setter for the power at the heart of this distribution.
        
        value: a positive number
        """
        if type(value) in real_numerical_types:
            if value > 0:
                self._power = value
            else:
                raise ValueError("power was not a positive number.")
        else:
            raise TypeError("power was not a real number.")
    
    @property
    def log_abs_power(self):
        """
        Property storing the natural logarithm of the power property.
        """
        if not hasattr(self, '_log_abs_power'):
            self._log_abs_power = np.log(np.abs(self.power))
        return self._log_abs_power
    
    def derivative(self, value):
        """
        Computes the derivative of the function underlying this Transform at
        the given value(s).
        
        value: single number or numpy.ndarray of values
        
        returns: value of derivative in same format as value
        """
        return self.power * np.power(value, self.power - 1)
    
    def second_derivative(self, value):
        """
        Computes the second derivative of the function underlying this
        Transform at the given value(s).
        
        value: single number or numpy.ndarray of values
        
        returns: value of second derivative in same format as value
        """
        return\
            (self.power * (self.power - 1) * np.power(value, self.power - 2))
    
    def third_derivative(self, value):
        """
        Computes the third derivative of the function underlying this Transform
        at the given value(s).
        
        value: single number or numpy.ndarray of values
        
        returns: value of third derivative in same format as value
        """
        return (self.power * (self.power - 1) * (self.power - 2) *\
            np.power(value, self.power - 3))
    
    def log_derivative(self, value):
        """
        Computes the natural logarithm of the derivative of the function
        underlying this Transform at the given value(s).
        
        value: single number or numpy.ndarray of values
        
        returns: value of log derivative in same format as value
        """
        return (self.log_abs_power + ((self.power - 1) * np.log(value)))
    
    def derivative_of_log_derivative(self, value):
        """
        Computes the derivative of the natural logarithm of the derivative of
        the function underlying this Transform at the given value(s).
        
        value: single number or numpy.ndarray of values
        
        returns: value of derivative of log derivative in same format as value
        """
        return ((self.power - 1) / value)
    
    def second_derivative_of_log_derivative(self, value):
        """
        Computes the second derivative of the natural logarithm of the
        derivative of the function underlying this Transform at the given
        value(s).
        
        value: single number or numpy.ndarray of values
        
        returns: value of second derivative of log derivative in same format as
                 value
        """
        return ((1 - self.power) / (value ** 2))
    
    def apply(self, value):
        """
        Applies this transform to the value and returns the result.
        
        value: single number or numpy.ndarray of values
        
        returns: value of function in same format as value
        """
        return np.power(value, self.power)
    
    def apply_inverse(self, value):
        """
        Applies the inverse of this transform to the value.
        
        value: single number or numpy.ndarray of values
        
        returns: value of inverse function in same format as value
        """
        return np.power(value, 1 / self.power)
    
    def __eq__(self, other):
        """
        Checks for equality with other. Returns True iff other is a
        PowerTransform with the same power.
        """
        if isinstance(other, PowerTransform):
            return (self.power == other.power)
        else:
            return False
    
    def to_string(self):
        """
        Generates a string version of this Transform.
        
        returns: value which can be cast into this Transform
        """
        return 'Power {:.2g}'.format(self.power)
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data about this transform.
        
        group: hdf5 file group to which to write data about this transform
        """
        group.attrs['class'] = 'PowerTransform'
        group.attrs['power'] = self.power

