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
    def log_value_addition(self, value):
        """
        Finds the term which should be added to the log value of the
        distribution due to this transform (pretty much the log of the
        derivative of the transformed parameter with respect to the original
        parameter evaluated at value).
        
        value: numerical variable value at which to evaluate things
        
        returns: single number to add to log value of distribution
        """
        raise NotImplementedError("Transform cannot be directly instantiated.")
    
    def apply(self, value):
        """
        Applies this transform to the value and returns the result.
        
        value: numerical variable value at which to evaluate things
        
        returns: transformed value
        """
        raise NotImplementedError("Transform cannot be directly instantiated.")
    
    def __call__(self, value):
        """
        Calling a transform object is simply an alias for the apply(value)
        function.
        
        value: number to transform
        
        returns: transformed value
        """
        return self.apply(value)
    
    def apply_inverse(self, value):
        """
        Applies the inverse of this transform to the value.
        
        value: numerical variable value at which to evaluate things
        
        returns: value which, when this transform is applied to it, gives value
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

