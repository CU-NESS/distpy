"""
File: distpy/transform/ArsinhTransform.py
Author: Keith Tauscher
Date: 2 Oct 2018

Description: File containing class representing transform which takes sinh or
             arcsinh value. It was originally given in:
             Shuhmann, R.L., Joachimi, B., Peiris, H.V., Gaussianization for
             fast and accurate inference from cosmological data, Monthly
             Notices of the Royal Astronomical Society, Volume 459, Issue 2,
             21 June 2016, Pages 1916-1928 https://doi.org/10.1093/mnras/stw738
"""
from __future__ import division
from ..util import real_numerical_types
import numpy as np
from .Transform import Transform

class ArsinhTransform(Transform):
    """
    Class representing a transform which takes sinh or arcsinh of value.
    """
    def __init__(self, shape):
        """
        Initializes a new BoxCoxTransform with the given shape.
        
        shape: single real number.
               Sinh used if shape>0, Arcsinh used if shape<0
        """
        self.shape = shape
    
    @property
    def shape(self):
        """
        Property determining the function which is taken of this transform's
        argument.
        """
        if not hasattr(self, '_shape'):
            raise AttributeError("shape was referenced before it was set.")
        return self._shape
    
    @shape.setter
    def shape(self, value):
        """
        Setter for the property determining the function which is taken of this
        transform's argument.
        
        value: single number, greater than or equal to 0
        """
        if type(value) in real_numerical_types:
            self._shape = value
        else:
            raise TypeError("shape was set to a non-number.")
    
    def derivative(self, value):
        """
        Computes the derivative of the function underlying this Transform at
        the given value(s).
        
        value: single number or numpy.ndarray of values
        
        returns: value of derivative in same format as value
        """
        if self.shape > 0:
            return np.cosh(self.shape * (value))
        elif self.shape < 0:
            return np.power(np.power(self.shape * (value), 2) +\
                1, -0.5)
        else:
            return np.power(value,  0)
    
    def second_derivative(self, value):
        """
        Computes the second derivative of the function underlying this
        Transform at the given value(s).
        
        value: single number or numpy.ndarray of values
        
        returns: value of second derivative in same format as value
        """
        if self.shape > 0:
            return self.shape * np.sinh(self.shape * (value))
        elif self.shape < 0:
            return (((self.shape ** 2) * ((-1) * value)) /\
                np.power(1 + ((self.shape * value) ** 2), 1.5))
        else:
            return value * 0
    
    def third_derivative(self, value):
        """
        Computes the third derivative of the function underlying this Transform
        at the given value(s).
        
        value: single number or numpy.ndarray of values
        
        returns: value of third derivative in same format as value
        """
        if self.shape > 0:
            return (self.shape ** 2) * np.cosh(self.shape * value)
        elif self.shape < 0:
            argument = np.power(self.shape * value, 2)
            return ((self.shape ** 2) * ((2 * argument) - 1)) /\
                np.power(1 + argument, 2.5)
        else:
            return value * 0
    
    def log_derivative(self, value):
        """
        Computes the natural logarithm of the derivative of the function
        underlying this Transform at the given value(s).
        
        value: single number or numpy.ndarray of values
        
        returns: value of log derivative in same format as value
        """
        if self.shape > 0:
            return np.log(np.cosh(self.shape * value))
        elif self.shape < 0:
            return np.log(1 + ((self.shape * value) ** 2)) / (-2)
        else:
            return value * 0
    
    def derivative_of_log_derivative(self, value):
        """
        Computes the derivative of the natural logarithm of the derivative of
        the function underlying this Transform at the given value(s).
        
        value: single number or numpy.ndarray of values
        
        returns: value of derivative of log derivative in same format as value
        """
        if self.shape > 0:
            return self.shape * np.tanh(self.shape * value)
        elif self.shape < 0:
            return (((self.shape ** 2) * ((-1) * value)) /\
                (1 + np.power(self.shape * value, 2)))
        else:
            return value * 0
    
    def second_derivative_of_log_derivative(self, value):
        """
        Computes the second derivative of the natural logarithm of the
        derivative of the function underlying this Transform at the given
        value(s).
        
        value: single number or numpy.ndarray of values
        
        returns: value of second derivative of log derivative in same format as
                 value
        """
        if self.shape > 0:
            return np.power(\
                self.shape / np.cosh(self.shape * value), 2)
        elif self.shape < 0:
            argument = np.power(self.shape * value, 2)
            return ((argument - 1) * np.power(self.shape / (1 + argument), 2))
        else:
            return 0 * value
    
    def apply(self, value):
        """
        Applies this transform to the value and returns the result.
        
        value: single number or numpy.ndarray of values
        
        returns: value of function in same format as value
        """
        if self.shape > 0:
            return np.sinh(self.shape * value) / self.shape
        elif self.shape < 0:
            return np.arcsinh(self.shape * value) / self.shape
        else:
            return value
    
    def apply_inverse(self, value):
        """
        Applies the inverse of this transform to the value.
        
        value: single number or numpy.ndarray of values
        
        returns: value of inverse function in same format as value
        """
        if self.shape > 0:
            return (np.arcsinh(self.shape * value) / self.shape)
        elif self.shape < 0:
            return (np.sinh(self.shape * value) / self.shape)
        else:
            return value
    
    def __eq__(self, other):
        """
        Checks for equality with other. Returns True iff other is an
        ArsinhTransform with the same parameters.
        """
        if isinstance(other, ArsinhTransform):
            return np.isclose(self.shape, other.shape, atol=1e-6, rtol=1e-6)
        else:
            return False
    
    def to_string(self):
        """
        Generates a string version of this Transform.
        
        returns: value which can be cast into this Transform
        """
        return 'Arsinh({0:.2g})'.format(self.shape)
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data about this transform.
        
        group: hdf5 file group to which to write data about this transform
        """
        group.attrs['class'] = 'ArsinhTransform'
        group.attrs['shape'] = self.shape

