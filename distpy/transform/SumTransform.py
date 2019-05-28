"""
File: distpy/transform/SumTransform.py
Author: Keith Tauscher
Date: 12 Feb 2018

Description: File containing class describing the sum of arbitrary
             transformations.
"""
import numpy as np
from ..util import sequence_types
from .Transform import Transform

class SumTransform(Transform):
    """
    Class representing the sum of transformations.
    """
    def __init__(self, first_transform, second_transform):
        """
        Initializes a new SumTransform with the given transforms.
        
        first_transform, second_transform: must be Transform objects
        """
        self.transforms = [first_transform, second_transform]
    
    @property
    def transforms(self):
        """
        Property storing the transforms which this is the sum of.
        """
        if not hasattr(self, '_transforms'):
            raise AttributeError("transforms referenced before it was set.")
        return self._transforms
    
    @transforms.setter
    def transforms(self, value):
        """
        Setter for the transforms which this is the sum of.
        
        value: must be sequence of 2 Transform objects
        """
        if type(value) in sequence_types:
            if len(value) == 2:
                if all([isinstance(element, Transform) for element in value]):
                    self._transforms = value
                else:
                    raise TypeError("At least one element of the " +\
                        "transforms sequence was not a Transform object.")
            else:
                raise ValueError("More than two transforms were given.")
        else:
            raise TypeError("transforms was set to a non-sequence.")
    
    def derivative(self, value):
        """
        Computes the derivative of the function underlying this Transform at
        the given value(s).
        
        value: single number or numpy.ndarray of values
        
        returns: value of derivative in same format as value
        """
        func_derivs =\
            [transform.derivative(value) for transform in self.transforms]
        return func_derivs[0] + func_derivs[1]
    
    def second_derivative(self, value):
        """
        Computes the second derivative of the function underlying this
        Transform at the given value(s).
        
        value: single number or numpy.ndarray of values
        
        returns: value of second derivative in same format as value
        """
        func_derivs2 = [transform.second_derivative(value)\
            for transform in self.transforms]
        return (func_derivs2[0] + func_derivs2[1])
    
    def third_derivative(self, value):
        """
        Computes the third derivative of the function underlying this Transform
        at the given value(s).
        
        value: single number or numpy.ndarray of values
        
        returns: value of third derivative in same format as value
        """
        func_derivs3 = [transform.third_derivative(value)\
            for transform in self.transforms]
        return func_derivs3[0] + func_derivs3[1]
    
    def log_derivative(self, value):
        """
        Computes the natural logarithm of the derivative of the function
        underlying this Transform at the given value(s).
        
        value: single number or numpy.ndarray of values
        
        returns: value of log derivative in same format as value
        """
        return np.log(np.abs(self.derivative(value)))
    
    def derivative_of_log_derivative(self, value):
        """
        Computes the derivative of the natural logarithm of the derivative of
        the function underlying this Transform at the given value(s).
        
        value: single number or numpy.ndarray of values
        
        returns: value of derivative of log derivative in same format as value
        """
        return self.second_derivative(value) / self.derivative(value)
    
    def second_derivative_of_log_derivative(self, value):
        """
        Computes the second derivative of the natural logarithm of the
        derivative of the function underlying this Transform at the given
        value(s).
        
        value: single number or numpy.ndarray of values
        
        returns: value of second derivative of log derivative in same format as
                 value
        """
        func_deriv = self.derivative(value)
        func_deriv2 = self.second_derivative(value)
        func_deriv3 = self.third_derivative(value)
        return (((func_deriv * func_deriv3) - (func_deriv2 ** 2)) /\
            (func_deriv ** 2))
    
    def apply(self, value):
        """
        Applies this transform to the value and returns the result.
        
        value: single number or numpy.ndarray of values
        
        returns: value of function in same format as value
        """
        return self.transforms[0](value) + self.transforms[1](value)
    
    def apply_inverse(self, value):
        """
        Applies the inverse of this transform to the value.
        
        value: single number or numpy.ndarray of values
        
        returns: value of inverse function in same format as value
        """
        raise NotImplementedError("The SumTransform cannot be inverted.")
    
    def to_string(self):
        """
        Generates a string version of this Transform.
        
        returns: value which can be cast into this Transform
        """
        return '({0!s}+{1!s})'.format(self.transforms[0].to_string(),\
            self.transforms[1].to_string())
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data about this transform.
        
        group: hdf5 file group to which to write data about this transform
        """
        group.attrs['class'] = 'SumTransform'
        self.transforms[0].fill_hdf5_group(group.create_group('transform_0'))
        self.transforms[1].fill_hdf5_group(group.create_group('transform_1'))
    
    def __eq__(self, other):
        """
        Fills the given hdf5 file group with data about this transform.
        
        other: object to check for equality
        
        returns True if both Transforms are the same
        """
        if isinstance(other, SumTransform):
            transforms_same = ((self.transforms[0] == other.transforms[0]) and\
                (self.transforms[1] == other.transforms[1]))
            if transforms_same:
                return True
            transforms_flipped =\
                ((self.transforms[0] == other.transforms[1]) and\
                (self.transforms[1] == other.transforms[0]))
            if transforms_flipped:
                return True
        return False

