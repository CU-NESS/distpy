"""
Module containing class representing a transformation of the form:
$$x\\longrightarrow e^x$$

**File**: $DISTPY/distpy/transform/ExponentialTransform.py  
**Author**: Keith Tauscher  
**Date**: 17 May 2021
"""
import numpy as np
from .Transform import Transform

class ExponentialTransform(Transform):
    """
    Class representing a transformation of the form: $$x\\longrightarrow e^x$$
    """
    def derivative(self, value):
        """
        Computes the derivative of the function underlying this
        `ExponentialTransform` at the given value(s).
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the derivative
        
        Returns
        -------
        derivative : number or sequence
            value of derivative of transformation in same format as `value`. If
            `value` is \\(x\\), then `derivative` is \\(e^x\\)
        """
        return np.exp(value)
    
    def second_derivative(self, value):
        """
        Computes the second derivative of the function underlying this
        `ExponentialTransform` at the given value(s).
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the derivative
        
        Returns
        -------
        derivative : number or sequence
            value of second derivative of transformation in same format as
            `value`. If `value` is \\(x\\), then `derivative` is \\(e^x\\)
        """
        return np.exp(value)
    
    def third_derivative(self, value):
        """
        Computes the third derivative of the function underlying this
        `ExponentialTransform` at the given value(s).
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the derivative
        
        Returns
        -------
        derivative : number or sequence
            value of third derivative of transformation in same format as
            `value`. If `value` is \\(x\\), then `derivative` is \\(e^x\\)
        """
        return np.exp(value)
    
    def log_derivative(self, value):
        """
        Computes the natural logarithm of the absolute value of the derivative
        of the function underlying this `ExponentialTransform` at the given
        value(s).
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the derivative
        
        Returns
        -------
        derivative : number or sequence
            value of the log of the derivative of transformation in same format
            as `value`. If `value` is \\(x\\), then `derivative` is \\(x\\)
        """
        return value
    
    def derivative_of_log_derivative(self, value):
        """
        Computes the derivative of the natural logarithm of the absolute value
        of the derivative of the function underlying this
        `ExponentialTransform` at the given value(s).
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the derivative
        
        Returns
        -------
        derivative : number or sequence
            value of the derivative of the log of the derivative of
            transformation in same format as `value`. If `value` is \\(x\\),
            then `derivative` is \\(1\\)
        """
        return (0. * value) + 1.
    
    def second_derivative_of_log_derivative(self, value):
        """
        Computes the second derivative of the natural logarithm of the absolute
        value of the derivative of the function underlying this
        `ExponentialTransform` at the given value(s).
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the derivative
        
        Returns
        -------
        derivative : number or sequence
            value of the second derivative of the log of the derivative of
            transformation in same format as `value`. If `value` is \\(x\\),
            then `derivative` is \\(0\\)
        """
        return 0. * value
    
    def apply(self, value):
        """
        Applies this `ExponentialTransform` to the value and returns the
        result.
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the
            transformation
        
        Returns
        -------
        transformed : number or sequence
            transformed value same format as `value`. If `value` is \\(x\\),
            then `transformed` is \\(e^x\\)
        """
        return np.exp(value)
    
    def apply_inverse(self, value):
        """
        Applies the inverse of this `ExponentialTransform` to the value and
        returns the result.
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the inverse
            transformation
        
        Returns
        -------
        inverted : number or sequence
            untransformed value same format as `value`. If `value` is \\(y\\),
            then `inverted` is \\(\\ln{(y)}\\)
        """
        return np.log(value)
    
    def __eq__(self, other):
        """
        Checks the given object for equality with this `ExponentialTransform`.
        
        Parameters
        ----------
        other : object
            object to check for equality
        
        Returns
        -------
        result : bool
            True if and only if `other` is another `ExponentialTransform`
        """
        return isinstance(other, ExponentialTransform)
    
    def to_string(self):
        """
        Generates a string version of this `ExponentialTransform`.
        
        Returns
        -------
        representation : str
            `'exp'`
        """
        return 'exp'
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data about this
        `ExponentialTransform` so it can be loaded later.
        
        Parameters
        ----------
        group : h5py.Group
            hdf5 file group to which to write data about this
            `ExponentialTransform`
        """
        group.attrs['class'] = 'ExponentialTransform'

