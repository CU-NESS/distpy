"""
Module containing class representing a transformation of the form:
$$x\\longrightarrow\\ln{(x)}$$

**File**: $DISTPY/distpy/transform/LogTransform.py  
**Author**: Keith Tauscher  
**Date**: 17 May 2021
"""
import numpy as np
from .Transform import Transform

class LogTransform(Transform):
    """
    Class representing a transformation of the form:
    $$x\\longrightarrow\\ln{(x)}$$
    """
    def derivative(self, value):
        """
        Computes the derivative of the function underlying this `LogTransform`
        at the given value(s).
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the derivative
        
        Returns
        -------
        derivative : number or sequence
            value of derivative of transformation in same format as `value`. If
            `value` is \\(x\\), then `derivative` is \\(\\frac{1}{x}\\)
        """
        return 1. / value
    
    def second_derivative(self, value):
        """
        Computes the second derivative of the function underlying this
        `LogTransform` at the given value(s).
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the derivative
        
        Returns
        -------
        derivative : number or sequence
            value of second derivative of transformation in same format as
            `value`. If `value` is \\(x\\), then `derivative` is
            \\(-\\frac{1}{x^2}\\)
        """
        return (-1.) / (value ** 2)
    
    def third_derivative(self, value):
        """
        Computes the third derivative of the function underlying this
        `LogTransform` at the given value(s).
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the derivative
        
        Returns
        -------
        derivative : number or sequence
            value of third derivative of transformation in same format as
            `value`. If `value` is \\(x\\), then `derivative` is
            \\(\\frac{2}{x^3}\\)
        """
        return 2. / (value ** 3)
    
    def log_derivative(self, value):
        """
        Computes the natural logarithm of the absolute value of the derivative
        of the function underlying this `LogTransform` at the given value(s).
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the derivative
        
        Returns
        -------
        derivative : number or sequence
            value of the log of the derivative of transformation in same format
            as `value`. If `value` is \\(x\\), then `derivative` is
            \\(-\\ln{(x)}\\)
        """
        return -np.log(value)
    
    def derivative_of_log_derivative(self, value):
        """
        Computes the derivative of the natural logarithm of the absolute value
        of the derivative of the function underlying this `LogTransform` at the
        given value(s).
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the derivative
        
        Returns
        -------
        derivative : number or sequence
            value of the derivative of the log of the derivative of
            transformation in same format as `value`. If `value` is \\(x\\),
            then `derivative` is \\(-\\frac{1}{x}\\)
        """
        return (-1.) / value
    
    def second_derivative_of_log_derivative(self, value):
        """
        Computes the second derivative of the natural logarithm of the absolute
        value of the derivative of the function underlying this `LogTransform`
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
            then `derivative` is \\(\\frac{1}{x^2}\\)
        """
        return np.power(value, -2)
    
    def apply(self, value):
        """
        Applies this `LogTransform` to the value and returns the result.
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the
            transformation
        
        Returns
        -------
        transformed : number or sequence
            transformed value same format as `value`. If `value` is \\(x\\),
            then `transformed` is \\(\\ln{(x)}\\)
        """
        return np.log(value)
    
    def apply_inverse(self, value):
        """
        Applies the inverse of this `LogTransform` to the value and returns the
        result.
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the inverse
            transformation
        
        Returns
        -------
        inverted : number or sequence
            untransformed value same format as `value`. If `value` is \\(y\\),
            then `inverted` is \\(e^y\\)
        """
        return np.exp(value)
    
    def __eq__(self, other):
        """
        Checks the given object for equality with this `LogTransform`.
        
        Parameters
        ----------
        other : object
            object to check for equality
        
        Returns
        -------
        result : bool
            True if and only if `other` is another `LogTransform`
        """
        return isinstance(other, LogTransform)
    
    def to_string(self):
        """
        Generates a string version of this `LogTransform`.
        
        Returns
        -------
        representation : str
            `'log'`
        """
        return 'log'
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data about this `LogTransform` so
        it can be loaded later.
        
        Parameters
        ----------
        group : h5py.Group
            hdf5 file group to which to write data about this `LogTransform`
        """
        group.attrs['class'] = 'LogTransform'

