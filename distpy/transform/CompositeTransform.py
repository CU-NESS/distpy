"""
File: distpy/transform/Transform.py
Author: Keith Tauscher
Date: 12 Feb 2018

Description: File containing class representing a transform composed of two
             separate transformations, an inner one (applied first) and an
             outer one (applied second).
"""
import numpy as np
from .Transform import Transform

class CompositeTransform(Transform):
    """
    Class representing a transformation which is composed of two separate
    transformations, an inner one (applied first) and an outer one (applied
    second).
    """
    def __init__(self, inner_transform, outer_transform):
        """
        Initializes a new CompositeTransform, h(x).
        
        inner_transform: if h(x)=f(g(x)), inner_transform is g
        outer_transform: if h(x)=f(g(x)), outer_transform is f
        """
        self.inner_transform = inner_transform
        self.outer_transform = outer_transform
    
    @property
    def inner_transform(self):
        """
        Property storing the innermost (first applied) transform composing this
        Transform.
        """
        if not hasattr(self, '_inner_transform'):
            raise AttributeError("inner_transform referenced before it was " +\
                "set.")
        return self._inner_transform
    
    @inner_transform.setter
    def inner_transform(self, value):
        """
        Setter for the innermost (first applied) transform composing this
        Transform.
        
        value: must be a Transform object
        """
        if isinstance(value, Transform):
            self._inner_transform = value
        else:
            raise TypeError("inner_transform given was not a Transform " +\
                "object.")
    
    @property
    def outer_transform(self):
        """
        Property storing the outermost (last applied) transform composing this
        Transform.
        """
        if not hasattr(self, '_outer_transform'):
            raise AttributeError("outer_transform referenced before it was " +\
                "set.")
        return self._outer_transform
    
    @outer_transform.setter
    def outer_transform(self, value):
        """
        Setter for the outermost (first applied) transform composing this
        Transform.
        
        value: must be a Transform object
        """
        if isinstance(value, Transform):
            self._outer_transform = value
        else:
            raise TypeError("outer_transform given was not a Transform " +\
                "object.")
    
    def derivative(self, value):
        """
        Computes the derivative of the function underlying this Transform at
        the given value(s).
        
        value: single number or numpy.ndarray of values
        
        returns: value of derivative in same format as value
        """
        inner_transformed_value = self.inner_transform(value)
        outer_derivative =\
            self.outer_transform.derivative(inner_transformed_value)
        inner_derivative = self.inner_transform.derivative(value)
        return outer_derivative * inner_derivative
    
    def second_derivative(self, value):
        """
        Computes the second derivative of the function underlying this
        Transform at the given value(s).
        
        value: single number or numpy.ndarray of values
        
        returns: value of second derivative in same format as value
        """
        inner_transformed_value = self.inner_transform(value)
        outer_first_derivative =\
            self.outer_transform.derivative(inner_transformed_value)
        outer_second_derivative =\
            self.outer_transform.second_derivative(inner_transformed_value)
        inner_first_derivative = self.inner_transform.derivative(value)
        inner_second_derivative = self.inner_transform.second_derivative(value)
        first_term = (outer_second_derivative * (inner_first_derivative ** 2))
        second_term = (outer_first_derivative * inner_second_derivative)
        return (first_term + second_term)
    
    def third_derivative(self, value):
        """
        Computes the third derivative of the function underlying this Transform
        at the given value(s).
        
        value: single number or numpy.ndarray of values
        
        returns: value of third derivative in same format as value
        """
        inner_transformed_value = self.inner_transform(value)
        outer_first_derivative =\
            self.outer_transform.derivative(inner_transformed_value)
        outer_second_derivative =\
            self.outer_transform.second_derivative(inner_transformed_value)
        outer_third_derivative =\
            self.outer_transform.third_derivative(inner_transformed_value)
        inner_first_derivative = self.inner_transform.derivative(value)
        inner_second_derivative = self.inner_transform.second_derivative(value)
        inner_third_derivative = self.inner_transform.third_derivative(value)
        first_term = (outer_third_derivative * (inner_first_derivative ** 3))
        second_term = (3 * (outer_second_derivative *\
            (inner_first_derivative * inner_second_derivative)))
        third_term = outer_first_derivative * inner_third_derivative
        return ((first_term + second_term) + third_term)
    
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
        first = self.derivative(value)
        second = self.second_derivative(value)
        third = self.third_derivative(value)
        return (((first * third) - (second ** 2)) / (first ** 2))
    
    def apply(self, value):
        """
        Applies this transform to the value and returns the result.
        
        value: single number or numpy.ndarray of values
        
        returns: value of function in same format as value
        """
        return self.outer_transform(self.inner_transform(value))
    
    def apply_inverse(self, value):
        """
        Applies the inverse of this transform to the value.
        
        value: single number or numpy.ndarray of values
        
        returns: value of inverse function in same format as value
        """
        return self.inner_transform.I(self.outer_transform.I(value))
    
    def to_string(self):
        """
        Generates a string version of this Transform.
        
        returns: value which can be cast into this Transform
        """
        return '({0!s} then {1!s})'.format(self.inner_transform.to_string,\
            self.outer_transform.to_string)
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data about this transform.
        
        group: hdf5 file group to which to write data about this transform
        """
        group.attrs['class'] = 'CompositeTransform'
        self.inner_transform.fill_hdf5_group(\
            group.create_group('inner_transform'))
        self.outer_transform.fill_hdf5_group(\
            group.create_group('outer_transform'))
    
    def __eq__(self, other):
        """
        Fills the given hdf5 file group with data about this transform.
        
        other: object to check for equality
        
        returns True if both Transforms are the same
        """
        if isinstance(other, CompositeTransform):
            inner_transforms_equal =\
                (self.inner_transform == other.inner_transform)
            outer_transforms_equal =\
                (self.outer_transform == other.outer_transform)
            return inner_transforms_equal and outer_transforms_equal
        else:
            return False

