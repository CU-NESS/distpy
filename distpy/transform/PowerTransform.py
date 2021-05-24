"""
Module containing class representing a transformation of the form:
$$x\\longrightarrow x^p$$

**File**: $DISTPY/distpy/transform/PowerTransform.py  
**Author**: Keith Tauscher  
**Date**: 17 May 2021
"""
from __future__ import division
import numpy as np
from ..util import real_numerical_types
from .Transform import Transform

class PowerTransform(Transform):
    """
    Class representing a transformation of the form: $$x\\longrightarrow x^p$$
    """
    def __init__(self, power):
        """
        Initializes a new `PowerTransform` which represents the following
        transformation: $$x\\longrightarrow x^p$$
        
        Parameters
        ----------
        power : number
            power to apply, \\(p\\)
        """
        self.power = power
    
    @property
    def power(self):
        """
        The power to raise inputs to, \\(p\\).
        """
        if not hasattr(self, '_power'):
            raise AttributeError("power was referenced before it was set.")
        return self._power
    
    @power.setter
    def power(self, value):
        """
        Setter for the `PowerTransform.power`.
        
        Parameters
        ----------
        value : number
            a non-zero number
        """
        if type(value) in real_numerical_types:
            if value > 0:
                self._power = value
            else:
                raise ValueError("power was not a positive number.")
        else:
            raise TypeError("power was not a real number.")
    
    @property
    def log_abs_power(self):
        """
        The natural logarithm of the absolute value of `PowerTransform.power`,
        \\(\\ln{|p|}\\).
        """
        if not hasattr(self, '_log_abs_power'):
            self._log_abs_power = np.log(np.abs(self.power))
        return self._log_abs_power
    
    def derivative(self, value):
        """
        Computes the derivative of the function underlying this
        `PowerTransform` at the given value(s).
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the derivative
        
        Returns
        -------
        derivative : number or sequence
            value of derivative of transformation in same format as `value`. If
            `value` is \\(x\\), then `derivative` is \\(px^{p-1}\\), where
            \\(p\\) is `PowerTransform.power`
        """
        return self.power * np.power(value, self.power - 1)
    
    def second_derivative(self, value):
        """
        Computes the second derivative of the function underlying this
        `PowerTransform` at the given value(s).
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the derivative
        
        Returns
        -------
        derivative : number or sequence
            value of second derivative of transformation in same format as
            `value`. If `value` is \\(x\\), then `derivative` is
            \\(p(p-1)x^{p-2}\\), where \\(p\\) is `PowerTransform.power`
        """
        return\
            (self.power * (self.power - 1) * np.power(value, self.power - 2))
    
    def third_derivative(self, value):
        """
        Computes the third derivative of the function underlying this
        `PowerTransform` at the given value(s).
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the derivative
        
        Returns
        -------
        derivative : number or sequence
            value of third derivative of transformation in same format as
            `value`. If `value` is \\(x\\), then `derivative` is
            \\(p(p-1)(p-2)x^{p-3}\\), where \\(p\\) is `PowerTransform.power`
        """
        return (self.power * (self.power - 1) * (self.power - 2) *\
            np.power(value, self.power - 3))
    
    def log_derivative(self, value):
        """
        Computes the natural logarithm of the absolute value of the derivative
        of the function underlying this `PowerTransform` at the given value(s).
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the derivative
        
        Returns
        -------
        derivative : number or sequence
            value of the log of the derivative of transformation in same format
            as `value`. If `value` is \\(x\\), then `derivative` is
            \\(\\ln{|p|} + (p-1)\\ln{(x)}\\), where \\(p\\) is
            `PowerTransform.power`
        """
        return (self.log_abs_power + ((self.power - 1) * np.log(value)))
    
    def derivative_of_log_derivative(self, value):
        """
        Computes the derivative of the natural logarithm of the absolute value
        of the derivative of the function underlying this `PowerTransform` at
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
            then `derivative` is \\(\\frac{p-1}{x}\\), where \\(p\\) is
            `PowerTransform.power`
        """
        return ((self.power - 1) / value)
    
    def second_derivative_of_log_derivative(self, value):
        """
        Computes the second derivative of the natural logarithm of the absolute
        value of the derivative of the function underlying this
        `AffineTransform` at the given value(s).
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the derivative
        
        Returns
        -------
        derivative : number or sequence
            value of the second derivative of the log of the derivative of
            transformation in same format as `value`. If `value` is \\(x\\),
            then `derivative` is \\(\\frac{1-p}{x^2}\\), where \\(p\\) is
            `PowerTransform.power`
        """
        return ((1 - self.power) / (value ** 2))
    
    def apply(self, value):
        """
        Applies this `PowerTransform` to the value and returns the result.
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the
            transformation
        
        Returns
        -------
        transformed : number or sequence
            transformed value same format as `value`. If `value` is \\(x\\),
            then `transformed` is \\(x^p\\), where \\(p\\) is
            `PowerTransform.power`
        """
        return np.power(value, self.power)
    
    def apply_inverse(self, value):
        """
        Applies the inverse of this `PowerTransform` to the value and returns
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
            then `inverted` is \\(y^{1/p}\\), where \\(p\\) is
            `PowerTransform.power`
        """
        return np.power(value, 1 / self.power)
    
    def __eq__(self, other):
        """
        Checks the given object for equality with this `PowerTransform`.
        
        Parameters
        ----------
        other : object
            object to check for equality
        
        Returns
        -------
        result : bool
            True if and only if `other` is another `PowerTransform` with the
            same `PowerTransform.power`
        """
        if isinstance(other, PowerTransform):
            return (self.power == other.power)
        else:
            return False
    
    def to_string(self):
        """
        Generates a string version of this `PowerTransform`.
        
        Returns
        -------
        representation : str
            `'Power p'`, where `p` is `PowerTransform.power`
        """
        return 'Power {:.2g}'.format(self.power)
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data about this `PowerTransform`
        so it can be loaded later.
        
        Parameters
        ----------
        group : h5py.Group
            hdf5 file group to which to write data about this `PowerTransform`
        """
        group.attrs['class'] = 'PowerTransform'
        group.attrs['power'] = self.power

