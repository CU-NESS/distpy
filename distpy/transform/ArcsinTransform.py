"""
Module containing class representing a transformation of the form:
$$x\\longrightarrow \\arcsin{(x)}$$

**File**: $DISTPY/distpy/transform/ArcsinTransform.py  
**Author**: Keith Tauscher  
**Date**: 17 May 2021
"""
import numpy as np
from .Transform import Transform

class ArcsinTransform(Transform):
    """
    Class representing a transformation of the form:
    $$x\\longrightarrow \\arcsin{(x)}$$
    """
    def derivative(self, value):
        """
        Computes the derivative of the function underlying this
        `ArcsinTransform` at the given value(s).
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the derivative
        
        Returns
        -------
        derivative : number or sequence
            value of derivative of transformation in same format as `value`. If
            `value` is \\(x\\), then `derivative` is \\((1-x^2)^{-1/2}\\)
        """
        return np.power(1. - (value ** 2), -0.5)
    
    def second_derivative(self, value):
        """
        Computes the second derivative of the function underlying this
        `ArcsinTransform` at the given value(s).
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the derivative
        
        Returns
        -------
        derivative : number or sequence
            value of second derivative of transformation in same format as
            `value`. If `value` is \\(x\\), then `derivative` is
            \\(\\frac{x}{(1-x^2)^{3/2}}\\)
        """
        return value * np.power(1 - (value ** 2), -1.5)
    
    def third_derivative(self, value):
        """
        Computes the third derivative of the function underlying this
        `ArcsinTransform` at the given value(s).
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the derivative
        
        Returns
        -------
        derivative : number or sequence
            value of third derivative of transformation in same format as
            `value`. If `value` is \\(x\\), then `derivative` is
            \\(\\frac{1+2x^2}{(1-x^2)^{5/2}}\\)
        """
        value2 = (value ** 2)
        return ((2. * value2) + 1.) / np.power(1. - (value2), 2.5)
    
    def log_derivative(self, value):
        """
        Computes the natural logarithm of the absolute value of the derivative
        of the function underlying this `ArcsinTransform` at the given
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
            \\(-\\frac{1}{2}\\ln{(1-x^2)}\\)
        """
        return -0.5 * np.log(1. - (value ** 2))
    
    def derivative_of_log_derivative(self, value):
        """
        Computes the derivative of the natural logarithm of the absolute value
        of the derivative of the function underlying this `ArcsinTransform` at
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
            then `derivative` is \\(\\frac{x}{1-x^2}\\)
        """
        return (value / (1. - (value ** 2)))
    
    def second_derivative_of_log_derivative(self, value):
        """
        Computes the second derivative of the natural logarithm of the absolute
        value of the derivative of the function underlying this
        `ArcsinTransform` at the given value(s).
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the derivative
        
        Returns
        -------
        derivative : number or sequence
            value of the second derivative of the log of the derivative of
            transformation in same format as `value`. If `value` is \\(x\\),
            then `derivative` is \\(\\frac{1+x^2}{(1-x^2)^2}\\)
        """
        value2 = (value ** 2)
        return ((1. + value2) / ((1. - value2) ** 2))
    
    def apply(self, value):
        """
        Applies this `ArcsinTransform` to the value and returns the result.
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the
            transformation
        
        Returns
        -------
        transformed : number or sequence
            transformed value same format as `value`. If `value` is \\(x\\),
            then `transformed` is \\(\\arcsin{(x)}\\)
        """
        return np.arcsin(value)
    
    def apply_inverse(self, value):
        """
        Applies the inverse of this `ArcsinTransform` to the value and returns
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
        return np.sin(value)
    
    def __eq__(self, other):
        """
        Checks the given object for equality with this `ArcsinTransform`.
        
        Parameters
        ----------
        other : object
            object to check for equality
        
        Returns
        -------
        result : bool
            True if and only if `other` is another `ArcsinTransform`
        """
        return isinstance(other, ArcsinTransform)
    
    def to_string(self):
        """
        Generates a string version of this `ArcsinTransform`.
        
        Returns
        -------
        representation : str
            `'arcsin'`
        """
        return 'arcsin'
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data about this `ArcsinTransform`
        so it can be loaded later.
        
        Parameters
        ----------
        group : h5py.Group
            hdf5 file group to which to write data about this `ArcsinTransform`
        """
        group.attrs['class'] = 'ArcsinTransform'

