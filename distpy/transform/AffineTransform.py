"""
Module containing class representing a transformation of the form:
$$x\\longrightarrow sx+t$$

**File**: $DISTPY/distpy/transform/AffineTransform.py  
**Author**: Keith Tauscher  
**Date**: 17 May 2021
"""
from __future__ import division
import numpy as np
from ..util import numerical_types
from .Transform import Transform

class AffineTransform(Transform):
    """
    Class representing a transformation of the form: $$x\\longrightarrow sx+t$$
    """
    def __init__(self, scale_factor, translation):
        """
        Initializes a new `AffineTransform` which represents the following
        transformation: $$x\\longrightarrow sx+t$$
        
        Parameters
        ----------
        scale_factor : number
            scaling factor, \\(s\\)
        translation : number
            translation, \\(t\\)
        """
        self.scale_factor = scale_factor
        self.translation = translation
    
    @property
    def scale_factor(self):
        """
        The factor by which inputs are multiplied, \\(s\\).
        """
        if not hasattr(self, '_scale_factor'):
            raise AttributeError("scale_factor referenced before it was set.")
        return self._scale_factor
    
    @scale_factor.setter
    def scale_factor(self, value):
        """
        Setter for the `AffineTransform.scale_factor`.
        
        Parameters
        ----------
        value : number
            a single real number
        """
        if type(value) in numerical_types:
            self._scale_factor = value
        else:
            raise TypeError("Can only scale by a single real number.")
    
    @property
    def translation(self):
        """
        The number added to the scaled input, \\(t\\).
        """
        if not hasattr(self, '_translation'):
            raise AttributeError("translation referenced before it was set.")
        return self._translation
    
    @translation.setter
    def translation(self, value):
        """
        Setter for `AffineTransform.translation`.
        
        Parameters
        ----------
        value : number
            a single real number
        """
        if type(value) in numerical_types:
            self._translation = value
        else:
            raise TypeError("Can only translate by a single real number.")
    
    def derivative(self, value):
        """
        Computes the derivative of the function underlying this
        `AffineTransform` at the given value(s).
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the derivative
        
        Returns
        -------
        derivative : number or sequence
            value of derivative of transformation in same format as `value`. If
            `value` is \\(x\\), then `derivative` is \\(s\\), where \\(s\\) is
            `AffineTransform.scale_factor`
        """
        return ((0. * value) + self.scale_factor)
    
    def second_derivative(self, value):
        """
        Computes the second derivative of the function underlying this
        `AffineTransform` at the given value(s).
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the derivative
        
        Returns
        -------
        derivative : number or sequence
            value of second derivative of transformation in same format as
            `value`. If `value` is \\(x\\), then `derivative` is \\(0\\)
        """
        return (0. * value)
    
    def third_derivative(self, value):
        """
        Computes the third derivative of the function underlying this
        `AffineTransform` at the given value(s).
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the derivative
        
        Returns
        -------
        derivative : number or sequence
            value of third derivative of transformation in same format as
            `value`. If `value` is \\(x\\), then `derivative` is \\(0\\)
        """
        return (0. * value)
    
    def log_derivative(self, value):
        """
        Computes the natural logarithm of the absolute value of the derivative
        of the function underlying this `AffineTransform` at the given value(s).
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the derivative
        
        Returns
        -------
        derivative : number or sequence
            value of the log of the derivative of transformation in same format
            as `value`. If `value` is \\(x\\), then `derivative` is \\(|s|\\),
            where \\(s\\) is `AffineTransform.scale_factor`
        """
        return ((0. * value) + np.log(np.abs(self.scale_factor)))
    
    def derivative_of_log_derivative(self, value):
        """
        Computes the derivative of the natural logarithm of the absolute value
        of the derivative of the function underlying this `AffineTransform` at
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
            then `derivative` is \\(0\\)
        """
        return (0. * value)
    
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
            then `derivative` is \\(0\\)
        """
        return (0. * value)
    
    def apply(self, value):
        """
        Applies this `AffineTransform` to the value and returns the result.
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the
            transformation
        
        Returns
        -------
        transformed : number or sequence
            transformed value same format as `value`. If `value` is \\(x\\),
            then `transformed` is \\(sx+t\\), where \\(s\\) and \\(t\\) are
            `AffineTransform.scale_factor` and `AffineTransform.translation`,
            respectively
        """
        return ((self.scale_factor * value) + self.translation)
    
    def apply_inverse(self, value):
        """
        Applies the inverse of this `AffineTransform` to the value and returns
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
            then `inverted` is \\((y-t)/s\\), where \\(s\\) and \\(t\\) are
            `AffineTransform.scale_factor` and `AffineTransform.translation`,
            respectively.
        """
        return ((value - self.translation) / self.scale_factor)
    
    def to_string(self):
        """
        Generates a string version of this `AffineTransform`.
        
        Returns
        -------
        representation : str
            `'Affine(s,t)'`, where `s` and `t` are
            `AffineTransform.scale_factor` and `AffineTransform.translation`,
            respectively
        """
        return 'Affine({0:.2g},{1:.2g})'.format(self.scale_factor,\
            self.translation)
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data about this `AffineTransform`
        so it can be loaded later.
        
        Parameters
        ----------
        group : h5py.Group
            hdf5 file group to which to write data about this `AffineTransform`
        """
        group.attrs['class'] = 'AffineTransform'
        group.attrs['scale_factor'] = self.scale_factor
        group.attrs['translation'] = self.translation
    
    def __eq__(self, other):
        """
        Checks the given object for equality with this `AffineTransform`.
        
        Parameters
        ----------
        other : object
            object to check for equality
        
        Returns
        -------
        result : bool
            True if and only if `other` is another `AffineTransform` with the
            same `AffineTransform.scale_factor` and
            `AffineTransform.translation`
        """
        if isinstance(other, AffineTransform):
            scale_factors_equal = np.isclose(self.scale_factor,\
                other.scale_factor, rtol=0, atol=1e-9)
            translations_equal = np.isclose(self.translation,\
                other.translation, rtol=1e-9, atol=0)
            return (scale_factors_equal and translations_equal)
        elif isinstance(other, NullTransform):
            return ((self.scale_factor == 1) and (self.translation == 0))
        else:
            return False

