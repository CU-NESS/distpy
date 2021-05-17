"""
Module containing class representing a transformation of the form:
$$x\\longrightarrow \\log_{10}{(x)}$$

**File**: $DISTPY/distpy/transform/Log10Transform.py  
**Author**: Keith Tauscher  
**Date**: 17 May 2021
"""
import numpy as np
from .Transform import Transform

ln10 = np.log(10)

class Log10Transform(Transform):
    """
    Class representing a transformation of the form:
    $$x\\longrightarrow \\log_{10}{(x)}$$
    """
    def derivative(self, value):
        """
        Computes the derivative of the function underlying this
        `Log10Transform` at the given value(s).
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the derivative
        
        Returns
        -------
        derivative : number or sequence
            value of derivative of transformation in same format as `value`. If
            `value` is \\(x\\), then `derivative` is
            \\(\\frac{1}{\\ln{10}\\times x}\\)
        """
        return 1. / (ln10 * value)
    
    def second_derivative(self, value):
        """
        Computes the second derivative of the function underlying this
        `Log10Transform` at the given value(s).
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the derivative
        
        Returns
        -------
        derivative : number or sequence
            value of second derivative of transformation in same format as
            `value`. If `value` is \\(x\\), then `derivative` is
            \\(-\\frac{1}{\\ln{10}\\times x^2}\\)
        """
        return (-1.) / (ln10 * (value ** 2))
    
    def third_derivative(self, value):
        """
        Computes the third derivative of the function underlying this
        `Log10Transform` at the given value(s).
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the derivative
        
        Returns
        -------
        derivative : number or sequence
            value of third derivative of transformation in same format as
            `value`. If `value` is \\(x\\), then `derivative` is
            \\(\\frac{2}{\\ln{10}\\times x^3}\\)
        """
        return 2. / (ln10 * (value ** 3))
    
    def log_derivative(self, value):
        """
        Computes the natural logarithm of the absolute value of the derivative
        of the function underlying this `Log10Transform` at the given value(s).
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the derivative
        
        Returns
        -------
        derivative : number or sequence
            value of the log of the derivative of transformation in same format
            as `value`. If `value` is \\(x\\), then `derivative` is
            \\(-\\ln{(\\ln{10}\\times x)}\\)
        """
        return -np.log(ln10 * value)
    
    def derivative_of_log_derivative(self, value):
        """
        Computes the derivative of the natural logarithm of the absolute value
        of the derivative of the function underlying this `Log10Transform` at
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
            then `derivative` is \\(-\\frac{1}{x}\\)
        """
        return (-1.) / value
    
    def second_derivative_of_log_derivative(self, value):
        """
        Computes the second derivative of the natural logarithm of the absolute
        value of the derivative of the function underlying this
        `Log10Transform` at the given value(s).
        
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
        return 1. / (value ** 2)
    
    def apply(self, value):
        """
        Applies this `Log10Transform` to the value and returns the result.
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the
            transformation
        
        Returns
        -------
        transformed : number or sequence
            transformed value same format as `value`. If `value` is \\(x\\),
            then `transformed` is \\(\\log_{10}{(x)}\\)
        """
        return np.log10(value)
    
    def apply_inverse(self, value):
        """
        Applies the inverse of this `Log10Transform` to the value and returns
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
            then `inverted` is \\(10^y\\)
        """
        return np.power(10., value)
    
    def __eq__(self, other):
        """
        Checks the given object for equality with this `Log10Transform`.
        
        Parameters
        ----------
        other : object
            object to check for equality
        
        Returns
        -------
        result : bool
            True if and only if `other` is another `Log10Transform`
        """
        return isinstance(other, Log10Transform)
    
    def to_string(self):
        """
        Generates a string version of this `Log10Transform`.
        
        Returns
        -------
        representation : str
            `'log10'`
        """
        return 'log10'
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data about this `Log10Transform`
        so it can be loaded later.
        
        Parameters
        ----------
        group : h5py.Group
            hdf5 file group to which to write data about this `Log10Transform`
        """
        group.attrs['class'] = 'Log10Transform'

