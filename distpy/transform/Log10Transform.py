"""
File: distpy/transform/Log10Transform.py
Author: Keith Tauscher
Date: 10 Sep 2017

Description: File containing class representing a Transform which performs
             logarithms base 10 on its inputs.
"""
import numpy as np
from .Transform import Transform

ln10 = np.log(10)

class Log10Transform(Transform):
    """
    Class representing a transform based on the base-10 logarithm function.
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
        return -np.log(ln10 * value)
    
    def apply(self, value):
        """
        Applies this transform to the value and returns the result.
        
        value: numerical variable value at which to evaluate things
        
        returns: transformed value
        """
        return np.log10(value)
    
    def apply_inverse(self, value):
        """
        Applies the inverse of this transform to the value.
        
        value: numerical variable value at which to evaluate things
        
        returns: value which, when this transform is applied to it, gives value
        """
        return np.power(10., value)
    
    def __eq__(self, other):
        """
        Checks for equality with other. Returns True iff other is a
        Log10Transform.
        """
        return isinstance(other, Log10Transform)
    
    def to_string(self):
        """
        Generates a string version of this Transform.
        
        returns: value which can be cast into this Transform
        """
        return 'log10'
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data about this transform.
        
        group: hdf5 file group to which to write data about this transform
        """
        group.attrs['class'] = 'Log10Transform'

