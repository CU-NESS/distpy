"""
Module containing class representing a transformation of the form:
$$x\\longrightarrow \\ln{\\left(\\frac{x}{1-x}\\right)}$$

**File**: $DISTPY/distpy/transform/LogisticTransform.py  
**Author**: Keith Tauscher  
**Date**: 17 May 2021
"""
import numpy as np
from .Transform import Transform

class LogisticTransform(Transform):
    """
    Class representing a transformation of the form:
    $$x\\longrightarrow \\ln{\\left(\\frac{x}{1-x}\\right)}$$
    """
    def derivative(self, value):
        """
        Computes the derivative of the function underlying this
        `LogisticTransform` at the given value(s).
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the derivative
        
        Returns
        -------
        derivative : number or sequence
            value of derivative of transformation in same format as `value`. If
            `value` is \\(x\\), then `derivative` is \\(\\frac{1}{x(1-x)}\\)
        """
        return 1. / (value * (1. - value))
    
    def second_derivative(self, value):
        """
        Computes the second derivative of the function underlying this
        `LogisticTransform` at the given value(s).
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the derivative
        
        Returns
        -------
        derivative : number or sequence
            value of second derivative of transformation in same format as
            `value`. If `value` is \\(x\\), then `derivative` is
            \\(\\frac{2x-1}{x^2(1-x)^2}\\)
        """
        return ((2. * value) - 1.) / ((value * (1. - value)) ** 2)
    
    def third_derivative(self, value):
        """
        Computes the third derivative of the function underlying this
        `LogisticTransform` at the given value(s).
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the derivative
        
        Returns
        -------
        derivative : number or sequence
            value of third derivative of transformation in same format as
            `value`. If `value` is \\(x\\), then `derivative` is
            \\(2\\left(\\frac{1}{x^3} + \\frac{1}{(1-x)^3}\\right)\\)
        """
        return 2. * (np.power(value, -3) + np.power(1. - value, -3))
    
    def log_derivative(self, value):
        """
        Computes the natural logarithm of the absolute value of the derivative
        of the function underlying this `LogisticTransform` at the given
        value(s).
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the derivative
        
        Returns
        -------
        derivative : number or sequence
            value of the log of the derivative of transformation in same format
            as `value`. If `value` is \\(x\\), then `derivative` is
            \\(-\\ln{(x)} - \\ln{(1-x)}\\)
        """
        return -np.log(value * (1. - value))
    
    def derivative_of_log_derivative(self, value):
        """
        Computes the derivative of the natural logarithm of the absolute value
        of the derivative of the function underlying this `LogisticTransform`
        at the given value(s).
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the derivative
        
        Returns
        -------
        derivative : number or sequence
            value of the derivative of the log of the derivative of
            transformation in same format as `value`. If `value` is \\(x\\),
            then `derivative` is \\(\\frac{1}{1-x}-\\frac{1}{x}\\)
        """
        return (1. / (1. - value)) - (1. / value)
    
    def second_derivative_of_log_derivative(self, value):
        """
        Computes the second derivative of the natural logarithm of the absolute
        value of the derivative of the function underlying this
        `LogisticTransform` at the given value(s).
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the derivative
        
        Returns
        -------
        derivative : number or sequence
            value of the second derivative of the log of the derivative of
            transformation in same format as `value`. If `value` is \\(x\\),
            then `derivative` is \\(\\frac{1}{x^2} + \\frac{1}{(1-x)^2}\\)
        """
        return ((1. / (value ** 2)) + (1. / ((1. - value) ** 2)))
    
    def apply(self, value):
        """
        Applies this `LogisticTransform` to the value and returns the result.
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the
            transformation
        
        Returns
        -------
        transformed : number or sequence
            transformed value same format as `value`. If `value` is \\(x\\),
            then `transformed` is \\(\\ln{\\left(\\frac{x}{1-x}\\right)}\\)
        """
        return np.log(value / (1. - value))
    
    def apply_inverse(self, value):
        """
        Applies the inverse of this `LogisticTransform` to the value and
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
            then `inverted` is \\(\\frac{1}{1 + e^{-y}}\\)
        """
        return 1. / (1. + np.exp(-value))
    
    def __eq__(self, other):
        """
        Checks the given object for equality with this `LogisticTransform`.
        
        Parameters
        ----------
        other : object
            object to check for equality
        
        Returns
        -------
        result : bool
            True if and only if `other` is another `LogisticTransform`
        """
        return isinstance(other, LogisticTransform)
    
    def to_string(self):
        """
        Generates a string version of this `LogisticTransform`.
        
        Returns
        -------
        representation : str
            `'logistic'`
        """
        return 'logistic'
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data about this
        `LogisticTransform` so it can be loaded later.
        
        Parameters
        ----------
        group : h5py.Group
            hdf5 file group to which to write data about this
            `LogisticTransform`
        """
        group.attrs['class'] = 'LogisticTransform'

