"""
Module containing class representing a transformation of the form:
$$x\\longrightarrow \\begin{cases} x & s = 0 \\\\ \\sinh{(sx)}/s & s > 0 \\\\\
\\text{arcsinh}{(sx)} / s & s < 0 \\end{cases}$$

**File**: $DISTPY/distpy/transform/ArsinhTransform.py  
**Author**: Keith Tauscher  
**Date**: 17 May 2021
"""
from __future__ import division
from ..util import real_numerical_types
import numpy as np
from .Transform import Transform

class ArsinhTransform(Transform):
    """
    Class representing a transformation of the form: $$x\\longrightarrow\
    \\begin{cases} x & s = 0 \\\\ \\sinh{(sx)}/s & s > 0 \\\\\
    \\text{arcsinh}{(sx)} / s & s < 0 \\end{cases}$$
    """
    def __init__(self, shape):
        """
        Initializes a new `ArsinhTransform` which represents the following
        transformation: $$x\\longrightarrow \\begin{cases} x & s = 0 \\\\\
        \\sinh{(sx)}/s & s > 0 \\\\\
        \\text{arcsinh}{(sx)} / s & s < 0 \\end{cases}$$
        
        Parameters
        ----------
        shape : number
            parameter determining shape of transformation, \\(s\\)
        """
        self.shape = shape
    
    @property
    def shape(self):
        """
        The number determining the function which is applied to inputs.
        """
        if not hasattr(self, '_shape'):
            raise AttributeError("shape was referenced before it was set.")
        return self._shape
    
    @shape.setter
    def shape(self, value):
        """
        Setter for `ArsinhTransform.shape`
        
        Parameters
        ----------
        value : number
            non-negative number
        """
        if type(value) in real_numerical_types:
            self._shape = value
        else:
            raise TypeError("shape was set to a non-number.")
    
    def derivative(self, value):
        """
        Computes the derivative of the function underlying this
        `ArsinhTransform` at the given value(s).
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the derivative
        
        Returns
        -------
        derivative : number or sequence
            value of derivative of transformation in same format as `value`. If
            `value` is \\(x\\), then `derivative` is
            \\(\\begin{cases} 1 & s = 0 \\\\ \\cosh{(sx)} & s > 0 \\\\\
            \\big(1+(sx)^2\\big)^{-1/2} & s < 0 \\end{cases}\\), where \\(s\\)
            is `ArsinhTransform.shape`
        """
        if self.shape > 0:
            return np.cosh(self.shape * (value))
        elif self.shape < 0:
            return np.power(np.power(self.shape * (value), 2) +\
                1, -0.5)
        else:
            return np.power(value,  0)
    
    def second_derivative(self, value):
        """
        Computes the second derivative of the function underlying this
        `ArsinhTransform` at the given value(s).
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the derivative
        
        Returns
        -------
        derivative : number or sequence
            value of second derivative of transformation in same format as
            `value`. If `value` is \\(x\\), then `derivative` is
            \\(\\begin{cases} 0 & s = 0 \\\\ s\\ \\sinh{(sx)} & s > 0 \\\\\
            -s\\frac{(sx)}{\\big(1+(sx)^2\\big)^{3/2}} & s < 0 \\end{cases}\\),
            where \\(s\\) is `ArsinhTransform.shape`
        """
        if self.shape > 0:
            return self.shape * np.sinh(self.shape * (value))
        elif self.shape < 0:
            return (((self.shape ** 2) * ((-1) * value)) /\
                np.power(1 + ((self.shape * value) ** 2), 1.5))
        else:
            return value * 0
    
    def third_derivative(self, value):
        """
        Computes the third derivative of the function underlying this
        `ArsinhTransform` at the given value(s).
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the derivative
        
        Returns
        -------
        derivative : number or sequence
            value of third derivative of transformation in same format as
            `value`. If `value` is \\(x\\), then `derivative` is
            \\(\\begin{cases} 0 & s = 0 \\\\ s^2\\ \\cosh{(sx)} & s > 0 \\\\\
            s^2\\frac{2(sx)^2-1}{\\big(1+(sx)^2\\big)^{5/2}} & s < 0\
            \\end{cases}\\), where \\(s\\) is `ArsinhTransform.shape`
        """
        if self.shape > 0:
            return (self.shape ** 2) * np.cosh(self.shape * value)
        elif self.shape < 0:
            argument = np.power(self.shape * value, 2)
            return ((self.shape ** 2) * ((2 * argument) - 1)) /\
                np.power(1 + argument, 2.5)
        else:
            return value * 0
    
    def log_derivative(self, value):
        """
        Computes the natural logarithm of the absolute value of the derivative
        of the function underlying this `ArsinhTransform` at the given
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
            \\(\\begin{cases} 0 & s = 0 \\\\\
            \\ln{\\big(\\cosh{(sx)}\\big)} & s > 0 \\\\\
            -\\frac{1}{2}\\ln{\\big(1+(sx)^2\\big)} & s < 0 \\end{cases}\\),
            where \\(s\\) is `ArsinhTransform.shape`
        """
        if self.shape > 0:
            return np.log(np.cosh(self.shape * value))
        elif self.shape < 0:
            return np.log(1 + ((self.shape * value) ** 2)) / (-2)
        else:
            return value * 0
    
    def derivative_of_log_derivative(self, value):
        """
        Computes the derivative of the natural logarithm of the absolute value
        of the derivative of the function underlying this `ArsinhTransform` at
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
            then `derivative` is \\(\\begin{cases} 0 & s = 0 \\\\\
            s\\ \\tanh{(sx)} & s > 0 \\\\\
            -s\\frac{(sx)}{1+(sx)^2} & s < 0 \\end{cases}\\), where \\(s\\) is
            `ArsinhTransform.shape`
        """
        if self.shape > 0:
            return self.shape * np.tanh(self.shape * value)
        elif self.shape < 0:
            return (((self.shape ** 2) * ((-1) * value)) /\
                (1 + np.power(self.shape * value, 2)))
        else:
            return (value * 0)
    
    def second_derivative_of_log_derivative(self, value):
        """
        Computes the second derivative of the natural logarithm of the absolute
        value of the derivative of the function underlying this
        `ArsinhTransform` at the given value(s).
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the derivative
        
        Returns
        -------
        derivative : number or sequence
            value of the second derivative of the log of the derivative of
            transformation in same format as `value`. If `value` is \\(x\\),
            then `derivative` is \\(\\begin{cases} 0 & s = 0\\\\\
            s^2\\ \\text{sech}^2{(sx)} & s > 0 \\\\\
            s^2\\frac{(sx)^2-1}{\\big(1+(sx)^2\\big)^2} & s < 0\
            \\end{cases}\\), where \\(s\\) is `ArsinhTransform.shape`
        """
        if self.shape > 0:
            return np.power(\
                self.shape / np.cosh(self.shape * value), 2)
        elif self.shape < 0:
            argument = np.power(self.shape * value, 2)
            return ((argument - 1) * np.power(self.shape / (1 + argument), 2))
        else:
            return 0 * value
    
    def apply(self, value):
        """
        Applies this `ArsinhTransform` to the value and returns the result.
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the
            transformation
        
        Returns
        -------
        transformed : number or sequence
            transformed value same format as `value`. If `value` is \\(x\\),
            then `transformed` is \\(\\begin{cases} x & s = 0 \\\\\
            \\sinh{(sx)}/s & s > 0 \\\\\
            \\text{arcsinh}{(sx)}/s & s < 0 \\end{cases}\\), where \\(s\\) is
            `ArsinhTransform.shape`
        """
        if self.shape > 0:
            return np.sinh(self.shape * value) / self.shape
        elif self.shape < 0:
            return np.arcsinh(self.shape * value) / self.shape
        else:
            return value
    
    def apply_inverse(self, value):
        """
        Applies the inverse of this `ArsinhTransform` to the value and returns
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
            then `inverted` is \\(\\begin{cases} y & s = 0 \\\\\
            \\text{arcsinh}{(sy)}/s & s > 0 \\\\\
            \\sinh{(sx)}/s & s < 0 \\end{cases}\\), where \\(s\\) is
            `ArsinhTransform.shape`
        """
        if self.shape > 0:
            return (np.arcsinh(self.shape * value) / self.shape)
        elif self.shape < 0:
            return (np.sinh(self.shape * value) / self.shape)
        else:
            return value
    
    def __eq__(self, other):
        """
        Checks the given object for equality with this `ArsinhTransform`.
        
        Parameters
        ----------
        other : object
            object to check for equality
        
        Returns
        -------
        result : bool
            True if and only if `other` is another `ArsinhTransform` with the
            same `ArsinhTransform.shape`
        """
        if isinstance(other, ArsinhTransform):
            return np.isclose(self.shape, other.shape, atol=1e-6, rtol=1e-6)
        else:
            return False
    
    def to_string(self):
        """
        Generates a string version of this `ArsinhTransform`.
        
        Returns
        -------
        representation : str
            `'Arsinh(s)'`, where `s` is `ArsinhTransform.shape`
        """
        return 'Arsinh({0:.2g})'.format(self.shape)
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data about this `ArsinhTransform`
        so it can be loaded later.
        
        Parameters
        ----------
        group : h5py.Group
            hdf5 file group to which to write data about this `ArsinhTransform`
        """
        group.attrs['class'] = 'ArsinhTransform'
        group.attrs['shape'] = self.shape

