"""
File: distpy/transform/AffineTransform.py
Author: Keith Tauscher
Date: 12 Feb 2018

Description: File containing class representing an affine transformation
             (linear transformations and translations).
"""
from __future__ import division
import numpy as np
from ..util import numerical_types
from .Transform import Transform

class AffineTransform(Transform):
    """
    Class representing an affine transformation (linear transformations and
    translations).
    """
    def __init__(self, scale_factor, translation):
        """
        Initializes a new affine transformation with the given scale factor (A)
        and translation (b). The resulting transform will implement x -> Ax+b
        
        scale_factor: number by which to multiply inputs
        translation: number to add to scaled inputs
        """
        self.scale_factor = scale_factor
        self.translation = translation
    
    @property
    def scale_factor(self):
        """
        Property storing the factor by which inputs are multiplied to generate
        the final output.
        """
        if not hasattr(self, '_scale_factor'):
            raise AttributeError("scale_factor referenced before it was set.")
        return self._scale_factor
    
    @scale_factor.setter
    def scale_factor(self, value):
        """
        Setter for the scale factor.
        
        value: must be a single real number
        """
        if type(value) in numerical_types:
            self._scale_factor = value
        else:
            raise TypeError("Can only scale by a single real number.")
    
    @property
    def translation(self):
        """
        Property storing the vector which is added to scaled input to generate
        the final output.
        """
        if not hasattr(self, '_translation'):
            raise AttributeError("translation referenced before it was set.")
        return self._translation
    
    @translation.setter
    def translation(self, value):
        """
        Setter for the translation.
        
        value: must be a single real number
        """
        if type(value) in numerical_types:
            self._translation = value
        else:
            raise TypeError("Can only translate by a single real number.")
    
    def derivative(self, value):
        """
        Computes the derivative of the function underlying this Transform at
        the given value(s).
        
        value: single number or numpy.ndarray of values
        
        returns: value of derivative in same format as value
        """
        return ((0. * value) + self.scale_factor)
    
    def second_derivative(self, value):
        """
        Computes the second derivative of the function underlying this
        Transform at the given value(s).
        
        value: single number or numpy.ndarray of values
        
        returns: value of second derivative in same format as value
        """
        return (0. * value)
    
    def third_derivative(self, value):
        """
        Computes the third derivative of the function underlying this Transform
        at the given value(s).
        
        value: single number or numpy.ndarray of values
        
        returns: value of third derivative in same format as value
        """
        return (0. * value)
    
    def log_derivative(self, value):
        """
        Computes the natural logarithm of the derivative of the function
        underlying this Transform at the given value(s).
        
        value: single number or numpy.ndarray of values
        
        returns: value of log derivative in same format as value
        """
        return ((0. * value) + np.log(self.scale_factor))
    
    def derivative_of_log_derivative(self, value):
        """
        Computes the derivative of the natural logarithm of the derivative of
        the function underlying this Transform at the given value(s).
        
        value: single number or numpy.ndarray of values
        
        returns: value of derivative of log derivative in same format as value
        """
        return (0. * value)
    
    def second_derivative_of_log_derivative(self, value):
        """
        Computes the second derivative of the natural logarithm of the
        derivative of the function underlying this Transform at the given
        value(s).
        
        value: single number or numpy.ndarray of values
        
        returns: value of second derivative of log derivative in same format as
                 value
        """
        return (0. * value)
    
    def apply(self, value):
        """
        Applies this transform to the value and returns the result.
        
        value: single number or numpy.ndarray of values
        
        returns: value of function in same format as value
        """
        return ((self.scale_factor * value) + self.translation)
    
    def apply_inverse(self, value):
        """
        Applies the inverse of this transform to the value.
        
        value: single number or numpy.ndarray of values
        
        returns: value of inverse function in same format as value
        """
        return ((value - self.translation) / self.scale_factor)
    
    def to_string(self):
        """
        Generates a string version of this Transform.
        
        returns: value which can be cast into this Transform
        """
        return 'Affine({0:.2g},{1:.2g})'.format(self.scale_factor,\
            self.translation)
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data about this transform.
        
        group: hdf5 file group to which to write data about this transform
        """
        group.attrs['class'] = 'AffineTransform'
        group.attrs['scale_factor'] = self.scale_factor
        group.attrs['translation'] = self.translation
    
    def __eq__(self, other):
        """
        Fills the given hdf5 file group with data about this transform.
        
        other: object to check for equality
        
        returns True if both Transforms are the same
        """
        if isinstance(other, AffineTransform):
            scale_factors_equal = np.isclose(self.scale_factor,\
                other.scale_factor, rtol=0, atol=1e-9)
            translations_equal = np.isclose(self.translation,\
                other.translation, rtol=1e-9, atol=0)
            return (scale_factors_equal and translations_equal)
        elif isinstance(other, NullTransform):
            return ((self.scale_factor == 1) and (self.translation == 0))
        else:
            return False

