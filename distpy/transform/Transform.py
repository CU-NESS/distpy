"""
File: distpy/transform/Transform.py
Author: Keith Tauscher
Date: 10 Sep 2017

Description: File containing base class for all built-in transformations.
"""
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

