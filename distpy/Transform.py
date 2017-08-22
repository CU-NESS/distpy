"""
File: distpy/Transform.py
Author: Keith Tauscher
Date: 17 Aug 2017

Description: File containing classes representing various common
             transformations. The transforms included are given below:
             
(1) NullTransform: applies no transform to value
(2) LogTransform: applies natural logarithm to value
(3) Log10Transform: applies logarithm base-10 to value
(4) SquareTransform: squares value
(5) ArcinTransform: applies inverse sine to value
(6) LogisticTransform: applies logistic function to value
"""
import numpy as np
from .Saving import Savable

valid_transforms = ['log', 'log10', 'square', 'arcsin', 'logistic']
ln10 = np.log(10)

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
    
    def apply_inverse(self, value):
        """
        Applies the inverse of this transform to the value.
        
        value: numerical variable value at which to evaluate things
        
        returns: value which, when this transform is applied to it, gives value
        """
        raise NotImplementedError("Transform cannot be directly instantiated.")
    
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

class NullTransform(Transform):
    """
    Class representing a transformation that doesn't actually transform.
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
        return 0.
    
    def apply(self, value):
        """
        Applies this transform to the value and returns the result.
        
        value: numerical variable value at which to evaluate things
        
        returns: transformed value
        """
        return value
    
    def apply_inverse(self, value):
        """
        Applies the inverse of this transform to the value.
        
        value: numerical variable value at which to evaluate things
        
        returns: value which, when this transform is applied to it, gives value
        """
        return value
    
    def __eq__(self, other):
        """
        Checks for equality with other. Returns True iff other is a
        NullTransform.
        """
        return isinstance(other, NullTransform)
    
    def to_string(self):
        """
        Generates a string version of this Transform.
        
        returns: value which can be cast into this Transform
        """
        return 'none'
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data about this transform.
        
        group: hdf5 file group to which to write data about this transform
        """
        group.attrs['class'] = 'NullTransform'

class LogTransform(Transform):
    """
    Class representing a transform based on the natural logarithm function.
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
        return -np.log(value)
    
    def apply(self, value):
        """
        Applies this transform to the value and returns the result.
        
        value: numerical variable value at which to evaluate things
        
        returns: transformed value
        """
        return np.log(value)
    
    def apply_inverse(self, value):
        """
        Applies the inverse of this transform to the value.
        
        value: numerical variable value at which to evaluate things
        
        returns: value which, when this transform is applied to it, gives value
        """
        return np.exp(value)
    
    def __eq__(self, other):
        """
        Checks for equality with other. Returns True iff other is a
        LogTransform.
        """
        return isinstance(other, LogTransform)
    
    def to_string(self):
        """
        Generates a string version of this Transform.
        
        returns: value which can be cast into this Transform
        """
        return 'log'
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data about this transform.
        
        group: hdf5 file group to which to write data about this transform
        """
        group.attrs['class'] = 'LogTransform'

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

class SquareTransform(Transform):
    """
    Class representing a transform based on the square function.
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
        return np.log(2 * value)
    
    def apply(self, value):
        """
        Applies this transform to the value and returns the result.
        
        value: numerical variable value at which to evaluate things
        
        returns: transformed value
        """
        return np.power(value, 2)
    
    def apply_inverse(self, value):
        """
        Applies the inverse of this transform to the value.
        
        value: numerical variable value at which to evaluate things
        
        returns: value which, when this transform is applied to it, gives value
        """
        return np.sqrt(value)
    
    def __eq__(self, other):
        """
        Checks for equality with other. Returns True iff other is a
        SquareTransform.
        """
        return isinstance(other, SquareTransform)
    
    def to_string(self):
        """
        Generates a string version of this Transform.
        
        returns: value which can be cast into this Transform
        """
        return 'square'
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data about this transform.
        
        group: hdf5 file group to which to write data about this transform
        """
        group.attrs['class'] = 'SquareTransform'

class ArcsinTransform(Transform):
    """
    Class representing a transform based on the inverse sine function.
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
        return -0.5 * np.log(1. - np.power(value, 2))
    
    def apply(self, value):
        """
        Applies this transform to the value and returns the result.
        
        value: numerical variable value at which to evaluate things
        
        returns: transformed value
        """
        return np.arcsin(value)
    
    def apply_inverse(self, value):
        """
        Applies the inverse of this transform to the value.
        
        value: numerical variable value at which to evaluate things
        
        returns: value which, when this transform is applied to it, gives value
        """
        return np.sin(value)
    
    def __eq__(self, other):
        """
        Checks for equality with other. Returns True iff other is a
        ArcsinTransform.
        """
        return isinstance(other, ArcsinTransform)
    
    def to_string(self):
        """
        Generates a string version of this Transform.
        
        returns: value which can be cast into this Transform
        """
        return 'arcsin'
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data about this transform.
        
        group: hdf5 file group to which to write data about this transform
        """
        group.attrs['class'] = 'ArcsinTransform'

class LogisticTransform(Transform):
    """
    Class representing a transform based on the logistic function.
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
        return -np.log(value * (1. - value))
    
    def apply(self, value):
        """
        Applies this transform to the value and returns the result.
        
        value: numerical variable value at which to evaluate things
        
        returns: transformed value
        """
        return np.log(value / (1. - value))
    
    def apply_inverse(self, value):
        """
        Applies the inverse of this transform to the value.
        
        value: numerical variable value at which to evaluate things
        
        returns: value which, when this transform is applied to it, gives value
        """
        return 1. / (1. + np.exp(-value))
    
    def __eq__(self, other):
        """
        Checks for equality with other. Returns True iff other is a
        LogisticTransform.
        """
        return isinstance(other, LogisticTransform)
    
    def to_string(self):
        """
        Generates a string version of this Transform.
        
        returns: value which can be cast into this Transform
        """
        return 'logistic'
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data about this transform.
        
        group: hdf5 file group to which to write data about this transform
        """
        group.attrs['class'] = 'LogisticTransform'


def cast_to_transform(key):
    """
    Loads a Transform from the given string key.
    
    key: either (1) None, (2) a string key from specifying which transform to
         load, or (3) a Transform object which will be parroted back
    
    returns: Transform object of the correct type
    """
    if key is None:
        return NullTransform()
    elif isinstance(key, str):
        lower_cased_key = key.lower()
        if lower_cased_key in ['null', 'none']:
            return NullTransform()
        elif lower_cased_key in ['log', 'ln']:
            return LogTransform()
        elif lower_cased_key == 'log10':
            return Log10Transform()
        elif lower_cased_key == 'square':
            return SquareTransform()
        elif lower_cased_key == 'arcsin':
            return ArcsinTransform()
        elif lower_cased_key == 'logistic':
            return LogisticTransform()
        else:
            raise ValueError("transform could not be reconstructed from " +\
                             "key, " + key + ", as key was not understood.")
    elif isinstance(key, Transform):
        return key
    else:
        raise TypeError("key cannot be cast to transform because it is " +\
                        "neither None nor a string nor a Transform.")

def castable_to_transform(key):
    try:
        cast_to_transform(key)
        return True
    except:
        return False

