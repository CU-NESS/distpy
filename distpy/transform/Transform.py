"""
File: distpy/transform/Transform.py
Author: Keith Tauscher
Date: 12 Feb 2018

Description: File containing base class for all built-in transformations.
"""
import numpy as np
from ..util import Savable

class Transform(Savable):
    """
    Class representing a transformation.
    """
    def derivative(self, value):
        """
        Computes the derivative of the function underlying this Transform at
        the given value(s).
        
        value: single number or numpy.ndarray of values
        
        returns: value of derivative in same format as value
        """
        raise NotImplementedError("Transform cannot be directly instantiated.")
    
    def second_derivative(self, value):
        """
        Computes the second derivative of the function underlying this
        Transform at the given value(s).
        
        value: single number or numpy.ndarray of values
        
        returns: value of second derivative in same format as value
        """
        raise NotImplementedError("Transform cannot be directly instantiated.")
    
    def third_derivative(self, value):
        """
        Computes the third derivative of the function underlying this Transform
        at the given value(s).
        
        value: single number or numpy.ndarray of values
        
        returns: value of third derivative in same format as value
        """
        raise NotImplementedError("Transform cannot be directly instantiated.")
    
    def log_derivative(self, value):
        """
        Computes the natural logarithm of the derivative of the function
        underlying this Transform at the given value(s).
        
        value: single number or numpy.ndarray of values
        
        returns: value of log derivative in same format as value
        """
        raise NotImplementedError("Transform cannot be directly instantiated.")
    
    def derivative_of_log_derivative(self, value):
        """
        Computes the derivative of the natural logarithm of the derivative of
        the function underlying this Transform at the given value(s).
        
        value: single number or numpy.ndarray of values
        
        returns: value of derivative of log derivative in same format as value
        """
        raise NotImplementedError("Transform cannot be directly instantiated.")
    
    def second_derivative_of_log_derivative(self, value):
        """
        Computes the second derivative of the natural logarithm of the
        derivative of the function underlying this Transform at the given
        value(s).
        
        value: single number or numpy.ndarray of values
        
        returns: value of second derivative of log derivative in same format as
                 value
        """
        raise NotImplementedError("Transform cannot be directly instantiated.")
    
    def apply(self, value):
        """
        Applies this transform to the value and returns the result.
        
        value: single number or numpy.ndarray of values
        
        returns: value of function in same format as value
        """
        raise NotImplementedError("Transform cannot be directly instantiated.")
    
    def __call__(self, value):
        """
        Calling a transform object is simply an alias for the apply(value)
        function.
        
        value: single number or numpy.ndarray of values
        
        returns: value of function in same format as value
        """
        return self.apply(value)
    
    def apply_inverse(self, value):
        """
        Applies the inverse of this transform to the value.
        
        value: single number or numpy.ndarray of values
        
        returns: value of inverse function in same format as value
        """
        raise NotImplementedError("Transform cannot be directly instantiated.")
    
    def I(self, value):
        """
        This function is an alias for the apply_inverse function.
        
        value: number to which to apply inverse transformation
        
        returns: value which, when this transform is applied to it, gives value
        """
        return self.apply_inverse(value)
    
    def untransform_minimum(self, transformed_minimum):
        """
        Untransforms the given minimum.
        
        transformed_minimum: the minimum in transformed space,
                             None represents -np.inf
        
        returns: if untransformed_minimum is finite, it is returned. Otherwise,
                 None is returned to indicate that the untransformed_minimum is
                 minus infinity
        """
        if type(transformed_minimum) is type(None):
            untransformed_minimum = self.apply_inverse(-np.inf)
        else:
            untransformed_minimum = self.apply_inverse(transformed_minimum)
        if np.isfinite(untransformed_minimum):
            return untransformed_minimum
        else:
            return None
    
    def untransform_maximum(self, transformed_maximum):
        """
        Untransforms the given maximum.
        
        transformed_maximum: the maximum in transformed space,
                             None represents +np.inf
        
        returns: if untransformed_maximum is finite, it is returned. Otherwise,
                 None is returned to indicate that the untransformed_maximum is
                 infinite
        """
        if type(transformed_maximum) is type(None):
            untransformed_maximum = self.apply_inverse(np.inf)
        else:
            untransformed_maximum = self.apply_inverse(transformed_maximum)
        if np.isfinite(untransformed_maximum):
            return untransformed_maximum
        else:
            return None
    
    def transform_minimum(self, untransformed_minimum):
        """
        Transforms the given minimum.
        
        untransformed_minimum: the minimum in untransformed space,
                               None represents -np.inf
        
        returns: if transformed_minimum is finite, it is returned. Otherwise,
                 None is returned to indicate that the transformed_minimum is
                 minus infinity
        """
        if type(untransformed_minimum) is type(None):
            transformed_minimum = self.apply(-np.inf)
        else:
            transformed_minimum = self.apply(untransformed_minimum)
        if np.isfinite(transformed_minimum):
            return transformed_minimum
        else:
            return None
    
    def transform_maximum(self, untransformed_maximum):
        """
        Transforms the given maximum.
        
        untransformed_maximum: the maximum in untransformed space,
                               None represents +np.inf
        
        returns: if transformed_maximum is finite, it is returned. Otherwise,
                 None is returned to indicate that the transformed_maximum is
                 infinite
        """
        if type(untransformed_maximum) is type(None):
            transformed_maximum = self.apply(np.inf)
        else:
            transformed_maximum = self.apply(untransformed_maximum)
        if np.isfinite(transformed_maximum):
            return transformed_maximum
        else:
            return None
    
    def to_string(self):
        """
        Generates a string version of this Transform.
        
        returns: value which can be cast into this Transform
        """
        raise NotImplementedError("Transform cannot be directly instantiated.")
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data about this transform.
        
        group: hdf5 file group to which to write data about this transform
        """
        raise NotImplementedError("Transform cannot be directly instantiated.")
    
    def __eq__(self, other):
        """
        Fills the given hdf5 file group with data about this transform.
        
        other: object to check for equality
        
        returns True if both Transforms are the same
        """
        raise NotImplementedError("Transform cannot be directly instantiated.")

    def __ne__(self, other):
        """
        Asserts that checks for equality are consistent with checks for
        inequality.
        
        other: object to check for inequality
        
        returns: opposite of __eq__
        """
        return (not self.__eq__(other))
    
    def __bool__(self):
        """
        This method makes it so that if-statements can be performed with
        variables storing Transforms as their expressions. If the variable
        contains a non-None Transform, it will return False.
        
        returns: True
        """
        return True

