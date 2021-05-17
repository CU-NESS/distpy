"""
Module containing class representing a transformation of the form:
$$x\\longrightarrow 10^x$$

**File**: $DISTPY/distpy/transform/Exp10Transform.py  
**Author**: Keith Tauscher  
**Date**: 17 May 2021
"""
import numpy as np
from .Transform import Transform

ln10 = np.log(10)
lnln10 = np.log(np.log(10))

class Exp10Transform(Transform):
    """
    Class representing a transformation of the form: $$x\\longrightarrow 10^x$$
    """
    def derivative(self, value):
        """
        Computes the derivative of the function underlying this
        `Exp10Transform` at the given value(s).
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the derivative
        
        Returns
        -------
        derivative : number or sequence
            value of derivative of transformation in same format as `value`. If
            `value` is \\(x\\), then `derivative` is \\(\\ln{10}\\times 10^x\\)
        """
        return (ln10 * np.power(10, value))
    
    def second_derivative(self, value):
        """
        Computes the second derivative of the function underlying this
        `Exp10Transform` at the given value(s).
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the derivative
        
        Returns
        -------
        derivative : number or sequence
            value of second derivative of transformation in same format as
            `value`. If `value` is \\(x\\), then `derivative` is
            \\((\\ln{10})^2\\ 10^x\\)
        """
        return ((ln10 ** 2) * np.power(10, value))
    
    def third_derivative(self, value):
        """
        Computes the third derivative of the function underlying this
        `Exp10Transform` at the given value(s).
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the derivative
        
        Returns
        -------
        derivative : number or sequence
            value of third derivative of transformation in same format as
            `value`. If `value` is \\(x\\), then `derivative` is
            \\((\\ln{10})^3\\ 10^x\\)
        """
        return ((ln10 ** 3) * np.power(10, value))
    
    def log_derivative(self, value):
        """
        Computes the natural logarithm of the absolute value of the derivative
        of the function underlying this `Exp10Transform` at the given value(s).
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the derivative
        
        Returns
        -------
        derivative : number or sequence
            value of the log of the derivative of transformation in same format
            as `value`. If `value` is \\(x\\), then `derivative` is
            \\(\\ln{(\\ln{10})}+(\\ln{10}\\times x)\\)
        """
        return (lnln10 + (ln10 * value))
    
    def derivative_of_log_derivative(self, value):
        """
        Computes the derivative of the natural logarithm of the absolute value
        of the derivative of the function underlying this `Exp10Transform` at
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
            then `derivative` is \\(\\ln{10}\\)
        """
        return (0 * value) + ln10
    
    def second_derivative_of_log_derivative(self, value):
        """
        Computes the second derivative of the natural logarithm of the absolute
        value of the derivative of the function underlying this
        `Exp10Transform` at the given value(s).
        
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
        Applies this `Exp10Transform` to the value and returns the result.
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the
            transformation
        
        Returns
        -------
        transformed : number or sequence
            transformed value same format as `value`. If `value` is \\(x\\),
            then `transformed` is \\(10^x\\)
        """
        return np.power(10, value)
    
    def apply_inverse(self, value):
        """
        Applies the inverse of this `Exp10Transform` to the value and returns
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
            then `inverted` is \\(\\log_{10}{(y)}\\)
        """
        return np.log10(value)
    
    def __eq__(self, other):
        """
        Checks the given object for equality with this `Exp10Transform`.
        
        Parameters
        ----------
        other : object
            object to check for equality
        
        Returns
        -------
        result : bool
            True if and only if `other` is another `Exp10Transform`
        """
        return isinstance(other, Exp10Transform)
    
    def to_string(self):
        """
        Generates a string version of this `Exp10Transform`.
        
        Returns
        -------
        representation : str
            `'exp10'`
        """
        return 'exp10'
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data about this `Exp10Transform`
        so it can be loaded later.
        
        Parameters
        ----------
        group : h5py.Group
            hdf5 file group to which to write data about this `Exp10Transform`
        """
        group.attrs['class'] = 'Exp10Transform'

