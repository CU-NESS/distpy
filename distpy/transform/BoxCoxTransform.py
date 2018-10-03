"""
File: distpy/transform/BoxCoxTransform.py
Author: Keith Tauscher
Date: 2 Oct 2018

Description: File containing class representing transform which takes powers of
             an offset version of its argument. It was originally given in:
             Robert L. Schuhmann, Benjamin Joachimi, Hiranya V. Peiris;
             Gaussianization for fast and accurate inference from cosmological
             data, Monthly Notices of the Royal Astronomical Society,
             Volume 459, Issue 2, 21 June 2016, Pages 1916–1928,
             https://doi.org/10.1093/mnras/stw738
"""
from __future__ import division
from ..util import real_numerical_types
import numpy as np
from .Transform import Transform

class BoxCoxTransform(Transform):
    """
    Class representing a transform based on powers of offset versions of its
    argument.
    """
    def __init__(self, power, offset=0):
        """
        Initializes a new BoxCoxTransform with the given power and offset.
        
        power: single number, greater than or equal to 0. This is the power
               taken of the offset version of this transform's argument
        offset: single real number. This is the quantity added to this
                transform's argument before the given power is taken
        """
        self.power = power
        self.offset = offset
    
    @property
    def power(self):
        """
        Property storing the power which is taken of the offset version of this
        transform's argument.
        """
        if not hasattr(self, '_power'):
            raise AttributeError("power was referenced before it was set.")
        return self._power
    
    @power.setter
    def power(self, value):
        """
        Setter for the power which is taken of the offset version of this
        transform's argument.
        
        value: single number, greater than or equal to 0
        """
        if type(value) in real_numerical_types:
            if value >= 0:
                self._power = value
            else:
                raise ValueError("power was negative.")
        else:
            raise TypeError("power was set to a non-number.")
    
    @property
    def offset(self):
        """
        Property storing the offset used in the transform. This is the quantity
        added to the untransformed value before a power of it is taken.
        """
        if not hasattr(self, '_offset'):
            raise AttributeError("offset was referenced before it was set.")
        return self._offset
    
    @offset.setter
    def offset(self, value):
        """
        Setter for the offset used in the transform. This is the quantity added
        to the untransformed value before a power is taken.
        
        value: single real number
        """
        if type(value) in real_numerical_types:
            self._offset = value
        else:
            raise TypeError("offset was not set to a number.")
    
    def derivative(self, value):
        """
        Computes the derivative of the function underlying this Transform at
        the given value(s).
        
        value: single number or numpy.ndarray of values
        
        returns: value of derivative in same format as value
        """
        return np.power(value + self.offset, self.power - 1)
    
    def second_derivative(self, value):
        """
        Computes the second derivative of the function underlying this
        Transform at the given value(s).
        
        value: single number or numpy.ndarray of values
        
        returns: value of second derivative in same format as value
        """
        return\
            ((self.power - 1) * np.power(value + self.offset, self.power - 2))
    
    def third_derivative(self, value):
        """
        Computes the third derivative of the function underlying this Transform
        at the given value(s).
        
        value: single number or numpy.ndarray of values
        
        returns: value of third derivative in same format as value
        """
        return ((self.power - 1) * (self.power - 2) *\
            np.power(value + self.offset, self.power - 3))
    
    def log_derivative(self, value):
        """
        Computes the natural logarithm of the derivative of the function
        underlying this Transform at the given value(s).
        
        value: single number or numpy.ndarray of values
        
        returns: value of log derivative in same format as value
        """
        return (self.power - 1) * np.log(value + self.offset)
    
    def derivative_of_log_derivative(self, value):
        """
        Computes the derivative of the natural logarithm of the derivative of
        the function underlying this Transform at the given value(s).
        
        value: single number or numpy.ndarray of values
        
        returns: value of derivative of log derivative in same format as value
        """
        return ((self.power - 1) / (value + self.offset))
    
    def second_derivative_of_log_derivative(self, value):
        """
        Computes the second derivative of the natural logarithm of the
        derivative of the function underlying this Transform at the given
        value(s).
        
        value: single number or numpy.ndarray of values
        
        returns: value of second derivative of log derivative in same format as
                 value
        """
        return ((1 - self.power) / np.power(value + self.offset, 2))
    
    def apply(self, value):
        """
        Applies this transform to the value and returns the result.
        
        value: single number or numpy.ndarray of values
        
        returns: value of function in same format as value
        """
        if self.power == 0:
            return np.log(value + self.offset)
        else:
            return (np.power(value + self.offset, self.power) - 1) / self.power
    
    def apply_inverse(self, value):
        """
        Applies the inverse of this transform to the value.
        
        value: single number or numpy.ndarray of values
        
        returns: value of inverse function in same format as value
        """
        if self.power == 0:
            return np.exp(value) - self.offset
        else:
            return np.power((self.power * value) + 1, 1 / self.power) -\
                self.offset
    
    def __eq__(self, other):
        """
        Checks for equality with other. Returns True iff other is a
        BoxCoxTransform with the same parameters.
        """
        if not isinstance(other, BoxCoxTransform):
            return False
        return np.allclose([self.power, self.offset],\
            [other.power, other.offset], atol=1e-6, rtol=1e-6)
    
    def to_string(self):
        """
        Generates a string version of this Transform.
        
        returns: value which can be cast into this Transform
        """
        return 'BoxCox({0:.2g},{1:.2g})'.format(self.power, self.offset)
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data about this transform.
        
        group: hdf5 file group to which to write data about this transform
        """
        group.attrs['class'] = 'BoxCoxTransform'
        group.attrs['power'] = self.power
        group.attrs['offset'] = self.offset

