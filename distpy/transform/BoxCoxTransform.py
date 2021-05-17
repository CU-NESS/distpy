"""
Module containing class representing a transformation of the form:
$$x\\longrightarrow \\begin{cases} \\ln{x+o} & p=0\\\\\
\\frac{(x+o)^p - 1}{p} & p \\ne 0 \\end{cases}$$ This transformation was
introduced in Shuhmann, R.L., Joachimi, B., Peiris, H.V., Gaussianization for
fast and accurate inference from cosmological data, Monthly Notices of the
Royal Astronomical Society, Volume 459, Issue 2, 21 June 2016, Pages 1916-1928
https://doi.org/10.1093/mnras/stw738

**File**: $DISTPY/distpy/transform/BoxCoxTransform.py  
**Author**: Keith Tauscher  
**Date**: 17 May 2021
"""
from __future__ import division
from ..util import real_numerical_types
import numpy as np
from .Transform import Transform

class BoxCoxTransform(Transform):
    """
    Class representing a transformation of the form:
    $$x\\longrightarrow \\begin{cases} \\ln{x+o} & p=0\\\\\
    \\frac{(x+o)^p - 1}{p} & p \\ne 0 \\end{cases}$$
    """
    def __init__(self, power, offset=0):
        """
        Initializes a new `BoxCoxTransform` which represents the following
        transformation: $$x\\longrightarrow \\begin{cases} \\ln{x+o} & p=0\\\\\
        \\frac{(x+o)^p - 1}{p} & p \\ne 0 \\end{cases}$$
        
        Parameters
        ----------
        power : number
            non-negative power to raise values to, \\(p\\)
        offset : number
            the negative of a sort of center, \\(o\\)
        """
        self.power = power
        self.offset = offset
    
    @property
    def power(self):
        """
        The power which is taken of the offset version of inputs.
        """
        if not hasattr(self, '_power'):
            raise AttributeError("power was referenced before it was set.")
        return self._power
    
    @power.setter
    def power(self, value):
        """
        Setter for `BoxCoxTransform.power`.
        
        Parameters
        ----------
        value : number
            non-negative number
        """
        if type(value) in real_numerical_types:
            if value >= 0:
                self._power = value
            else:
                raise ValueError("power was negative.")
        else:
            raise TypeError("power was set to a non-number.")
    
    @property
    def offset(self):
        """
        The offset applied to inputs.
        """
        if not hasattr(self, '_offset'):
            raise AttributeError("offset was referenced before it was set.")
        return self._offset
    
    @offset.setter
    def offset(self, value):
        """
        Setter for `BoxCoxTransform.offset`.
        
        Parameters
        ----------
        value : number
            real number
        """
        if type(value) in real_numerical_types:
            self._offset = value
        else:
            raise TypeError("offset was not set to a number.")
    
    def derivative(self, value):
        """
        Computes the derivative of the function underlying this
        `BoxCoxTransform` at the given value(s).
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the derivative
        
        Returns
        -------
        derivative : number or sequence
            value of derivative of transformation in same format as `value`. If
            `value` is \\(x\\), then `derivative` is \\((x+o)^{p-1}\\), where
            \\(o\\) and \\(p\\) are `BoxCoxTransform.offset` and
            `BoxCoxTransform.power`, respectively
        """
        return np.power(value + self.offset, self.power - 1)
    
    def second_derivative(self, value):
        """
        Computes the second derivative of the function underlying this
        `BoxCoxTransform` at the given value(s).
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the derivative
        
        Returns
        -------
        derivative : number or sequence
            value of second derivative of transformation in same format as
            `value`. If `value` is \\(x\\), then `derivative` is
            \\((p-1)(x+o)^{p-2}\\), where \\(o\\) and \\(p\\) are
            `BoxCoxTransform.offset` and `BoxCoxTransform.power`, respectively
        """
        return\
            ((self.power - 1) * np.power(value + self.offset, self.power - 2))
    
    def third_derivative(self, value):
        """
        Computes the third derivative of the function underlying this
        `BoxCoxTransform` at the given value(s).
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the derivative
        
        Returns
        -------
        derivative : number or sequence
            value of third derivative of transformation in same format as
            `value`. If `value` is \\(x\\), then `derivative` is
            \\((p-1)(p-2)(x+o)^{p-3}\\), where \\(o\\) and \\(p\\) are
            `BoxCoxTransform.offset` and `BoxCoxTransform.power`, respectively
        """
        return ((self.power - 1) * (self.power - 2) *\
            np.power(value + self.offset, self.power - 3))
    
    def log_derivative(self, value):
        """
        Computes the natural logarithm of the absolute value of the derivative
        of the function underlying this `BoxCoxTransform` at the given
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
            \\((p-1)\\ln{(x+o)}\\), where \\(o\\) and \\(p\\) are
            `BoxCoxTransform.offset` and `BoxCoxTransform.power`, respectively
        """
        return (self.power - 1) * np.log(value + self.offset)
    
    def derivative_of_log_derivative(self, value):
        """
        Computes the derivative of the natural logarithm of the absolute value
        of the derivative of the function underlying this `BoxCoxTransform` at
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
            then `derivative` is \\(\\frac{p-1}{x+o}\\), where \\(o\\) and
            \\(p\\) are `BoxCoxTransform.offset` and `BoxCoxTransform.power`,
            respectively
        """
        return ((self.power - 1) / (value + self.offset))
    
    def second_derivative_of_log_derivative(self, value):
        """
        Computes the second derivative of the natural logarithm of the absolute
        value of the derivative of the function underlying this
        `BoxCoxTransform` at the given value(s).
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the derivative
        
        Returns
        -------
        derivative : number or sequence
            value of the second derivative of the log of the derivative of
            transformation in same format as `value`. If `value` is \\(x\\),
            then `derivative` is \\(\\frac{1-p}{(x+o)^2}\\), where \\(o\\) and
            \\(p\\) are `BoxCoxTransform.offset` and `BoxCoxTransform.power`,
            respectively
        """
        return ((1 - self.power) / np.power(value + self.offset, 2))
    
    def apply(self, value):
        """
        Applies this `BoxCoxTransform` to the value and returns the result.
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the
            transformation
        
        Returns
        -------
        transformed : number or sequence
            transformed value same format as `value`. If `value` is \\(x\\),
            then `transformed` is \\(\\begin{cases} \\ln{(x+o)} & p = 0 \\\\\
            \\frac{(x+o)^p-1}{p} & p\\ne 0 \\end{cases}\\), where \\(o\\) and
            \\(p\\) are `BoxCoxTransform.offset` and `BoxCoxTransform.power`,
            respectively
        """
        if self.power == 0:
            return np.log(value + self.offset)
        else:
            return (np.power(value + self.offset, self.power) - 1) / self.power
    
    def apply_inverse(self, value):
        """
        Applies the inverse of this `BoxCoxTransform` to the value and returns
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
            then `inverted` is \\(\\begin{cases} e^y-o & p=0 \\\\\
            (py+1)^{1/p}-o & p\\ne 0 \\end{cases}\\), where \\(o\\) and \\(p\\)
            are `BoxCoxTransform.offset` and `BoxCoxTransform.power`,
            respectively
        """
        if self.power == 0:
            return np.exp(value) - self.offset
        else:
            return np.power((self.power * value) + 1, 1 / self.power) -\
                self.offset
    
    def __eq__(self, other):
        """
        Checks the given object for equality with this `BoxCoxTransform`.
        
        Parameters
        ----------
        other : object
            object to check for equality
        
        Returns
        -------
        result : bool
            True if and only if `other` is another `BoxCoxTransform` with the
            same `BoxCoxTransform.power` and `BoxCoxTransform.offset`
        """
        if not isinstance(other, BoxCoxTransform):
            return False
        return np.allclose([self.power, self.offset],\
            [other.power, other.offset], atol=1e-6, rtol=1e-6)
    
    def to_string(self):
        """
        Generates a string version of this `BoxCoxTransform`.
        
        Returns
        -------
        representation : str
            `'BoxCox(p,o)'`, where `p` and `o` are `BoxCoxTransform.power` and
            `BoxCoxTransform.offset`, respectively
        """
        return 'BoxCox({0:.2g},{1:.2g})'.format(self.power, self.offset)
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data about this `BoxCoxTransform`
        so it can be loaded later.
        
        Parameters
        ----------
        group : h5py.Group
            hdf5 file group to which to write data about this `BoxCoxTransform`
        """
        group.attrs['class'] = 'BoxCoxTransform'
        group.attrs['power'] = self.power
        group.attrs['offset'] = self.offset

