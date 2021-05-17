"""
Module containing class representing a transformation of the form:
$$x\\longrightarrow \\sin{(x)}$$

**File**: $DISTPY/distpy/transform/SineTransform.py  
**Author**: Keith Tauscher  
**Date**: 17 May 2021
"""
import numpy as np
from .Transform import Transform

class SineTransform(Transform):
    """
    Class representing a transformation of the form:
    $$x\\longrightarrow \\sin{(x)}$$
    """
    def derivative(self, value):
        """
        Computes the derivative of the function underlying this `SineTransform`
        at the given value(s).
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the derivative
        
        Returns
        -------
        derivative : number or sequence
            value of derivative of transformation in same format as `value`. If
            `value` is \\(x\\), then `derivative` is \\(\\cos{(x)}\\)
        """
        return np.cos(value)
    
    def second_derivative(self, value):
        """
        Computes the second derivative of the function underlying this
        `SineTransform` at the given value(s).
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the derivative
        
        Returns
        -------
        derivative : number or sequence
            value of second derivative of transformation in same format as
            `value`. If `value` is \\(x\\), then `derivative` is
            \\(-\\sin{(x)}\\)
        """
        return ((-1) * np.sin(value))
    
    def third_derivative(self, value):
        """
        Computes the third derivative of the function underlying this
        `SineTransform` at the given value(s).
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the derivative
        
        Returns
        -------
        derivative : number or sequence
            value of third derivative of transformation in same format as
            `value`. If `value` is \\(x\\), then `derivative` is
            \\(-\\cos{(x)}\\)
        """
        return ((-1) * np.cos(value))
    
    def log_derivative(self, value):
        """
        Computes the natural logarithm of the absolute value of the derivative
        of the function underlying this `SineTransform` at the given value(s).
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the derivative
        
        Returns
        -------
        derivative : number or sequence
            value of the log of the derivative of transformation in same format
            as `value`. If `value` is \\(x\\), then `derivative` is
            \\(\\ln{\\big(\\cos{(x)}\\big)}\\)
        """
        return np.log(np.cos(value))
    
    def derivative_of_log_derivative(self, value):
        """
        Computes the derivative of the natural logarithm of the absolute value
        of the derivative of the function underlying this `SineTransform` at
        the given value(s).
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the derivative
        
        Returns
        -------
        derivative : number or sequence
            value of the derivative of the log of the derivative of
            transformation in same format as `value`. If `value` is \\(x\\),
            then `derivative` is \\(-\\tan{(x)}\\)
        """
        return ((-1) * np.tan(value))
    
    def second_derivative_of_log_derivative(self, value):
        """
        Computes the second derivative of the natural logarithm of the absolute
        value of the derivative of the function underlying this `SineTransform`
        at the given value(s).
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the derivative
        
        Returns
        -------
        derivative : number or sequence
            value of the second derivative of the log of the derivative of
            transformation in same format as `value`. If `value` is \\(x\\),
            then `derivative` is \\(-\\sec^2{(x)}\\)
        """
        return ((-1) / (np.cos(value) ** 2))
    
    def apply(self, value):
        """
        Applies this `SineTransform` to the value and returns the result.
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the
            transformation
        
        Returns
        -------
        transformed : number or sequence
            transformed value same format as `value`. If `value` is \\(x\\),
            then `transformed` is \\(\\sin{(x)}\\)
        """
        return np.sin(value)
    
    def apply_inverse(self, value):
        """
        Applies the inverse of this `SineTransform` to the value and returns
        the result.
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the inverse
            transformation
        
        Returns
        -------
        inverted : number or sequence
            untransformed value same format as `value`. If `value` is \\(y\\),
            then `inverted` is \\(\\arcsin{(y)}\\)
        """
        return np.arcsin(value)
    
    def __eq__(self, other):
        """
        Checks the given object for equality with this `SineTransform`.
        
        Parameters
        ----------
        other : object
            object to check for equality
        
        Returns
        -------
        result : bool
            True if and only if `other` is another `SineTransform`
        """
        return isinstance(other, SineTransform)
    
    def to_string(self):
        """
        Generates a string version of this `SineTransform`.
        
        Returns
        -------
        representation : str
            `'sine'`
        """
        return 'sine'
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data about this `SineTransform` so
        it can be loaded later.
        
        Parameters
        ----------
        group : h5py.Group
            hdf5 file group to which to write data about this `SineTransform`
        """
        group.attrs['class'] = 'SineTransform'

