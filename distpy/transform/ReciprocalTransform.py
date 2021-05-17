"""
Module containing class representing a transformation of the form:
$$x\\longrightarrow \\frac{1}{x}$$

**File**: $DISTPY/distpy/transform/ReciprocalTransform.py  
**Author**: Keith Tauscher  
**Date**: 17 May 2021
"""
from __future__ import division
import numpy as np
from .Transform import Transform

class ReciprocalTransform(Transform):
    """
    Class representing a transformation of the form:
    $$x\\longrightarrow \\frac{1}{x}$$
    """
    def derivative(self, value):
        """
        Computes the derivative of the function underlying this
        `ReciprocalTransform` at the given value(s).
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the derivative
        
        Returns
        -------
        derivative : number or sequence
            value of derivative of transformation in same format as `value`. If
            `value` is \\(x\\), then `derivative` is \\(-\\frac{1}{x^2}\\)
        """
        return ((-1) / (value ** 2))
    
    def second_derivative(self, value):
        """
        Computes the second derivative of the function underlying this
        `ReciprocalTransform` at the given value(s).
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the derivative
        
        Returns
        -------
        derivative : number or sequence
            value of second derivative of transformation in same format as
            `value`. If `value` is \\(x\\), then `derivative` is
            \\(\\frac{2}{x^3}\\)
        """
        return (2 / (value ** 3))
    
    def third_derivative(self, value):
        """
        Computes the third derivative of the function underlying this
        `ReciprocalTransform` at the given value(s).
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the derivative
        
        Returns
        -------
        derivative : number or sequence
            value of third derivative of transformation in same format as
            `value`. If `value` is \\(x\\), then `derivative` is
            \\(-\\frac{6}{x^4}\\)
        """
        return ((-6) / (value ** 4))
    
    def log_derivative(self, value):
        """
        Computes the natural logarithm of the absolute value of the derivative
        of the function underlying this `ReciprocalTransform` at the given
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
            \\(-2\\times\\ln{|x|}\\)
        """
        return ((-2) * np.log(np.abs(value)))
    
    def derivative_of_log_derivative(self, value):
        """
        Computes the derivative of the natural logarithm of the absolute value
        of the derivative of the function underlying this `ReciprocalTransform`
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
            then `derivative` is \\(-\\frac{2}{x}\\)
        """
        return ((-2) / value)
    
    def second_derivative_of_log_derivative(self, value):
        """
        Computes the second derivative of the natural logarithm of the absolute
        value of the derivative of the function underlying this
        `ReciprocalTransform` at the given value(s).
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the derivative
        
        Returns
        -------
        derivative : number or sequence
            value of the second derivative of the log of the derivative of
            transformation in same format as `value`. If `value` is \\(x\\),
            then `derivative` is \\(\\frac{2}{x^2}\\)
        """
        return (2 / (value ** 2))
    
    def apply(self, value):
        """
        Applies this `ReciprocalTransform` to the value and returns the result.
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the
            transformation
        
        Returns
        -------
        transformed : number or sequence
            transformed value same format as `value`. If `value` is \\(x\\),
            then `transformed` is \\(\\frac{1}{x}\\)
        """
        return (1 / value)
    
    def apply_inverse(self, value):
        """
        Applies the inverse of this `ReciprocalTransform` to the value and
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
            then `inverted` is \\(\\frac{1}{y}\\)
        """
        return (1 / value)
    
    def to_string(self):
        """
        Generates a string version of this `ReciprocalTransform`.
        
        Returns
        -------
        representation : str
            `'reciprocal'`
        """
        return 'reciprocal'
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data about this
        `ReciprocalTransform` so it can be loaded later.
        
        Parameters
        ----------
        group : h5py.Group
            hdf5 file group to which to write data about this
            `ReciprocalTransform`
        """
        group.attrs['class'] = 'ReciprocalTransform'
    
    def __eq__(self, other):
        """
        Checks the given object for equality with this `ReciprocalTransform`.
        
        Parameters
        ----------
        other : object
            object to check for equality
        
        Returns
        -------
        result : bool
            True if and only if `other` is another `ReciprocalTransform`
        """
        return isinstance(other, ReciprocalTransform)

