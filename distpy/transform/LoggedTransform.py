"""
Module containing class representing a transformation of the form:
$$x\\longrightarrow \\ln{\\big(f(x)\\big)}$$

**File**: $DISTPY/distpy/transform/LoggedTransform.py  
**Author**: Keith Tauscher  
**Date**: 17 May 2021
"""
import numpy as np
from .Transform import Transform

class LoggedTransform(Transform):
    """
    Class representing a transformation of the form:
    $$x\\longrightarrow \\ln{\\big(f(x)\\big)}$$
    """
    def __init__(self, transform):
        """
        Initializes a new `LoggedTransform` which represents the following
        transformation: $$x\\longrightarrow \\ln{\\big(f(x)\\big)}$$
        
        Parameters
        ----------
        transform : `distpy.transform.Transform.Transform`
            transformation applied before taking the logarithm, \\(f\\)
        """
        self.transform = transform
    
    @property
    def transform(self):
        """
        The transform which this is the log of.
        """
        if not hasattr(self, '_transform'):
            raise AttributeError("transform referenced before it was set.")
        return self._transform
    
    @transform.setter
    def transform(self, value):
        """
        Setter for `ExponentiatedTransform.transform`.
        
        Parameters
        ----------
        value : `distpy.transform.Transform.Transform`
            the transformation to take the log of
        """
        if isinstance(value, Transform):
            self._transform = value
        else:
            raise TypeError("transform was not a Transform object.")
    
    def derivative(self, value):
        """
        Computes the derivative of the function underlying this
        `LoggedTransform` at the given value(s).
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the derivative
        
        Returns
        -------
        derivative : number or sequence
            value of derivative of transformation in same format as `value`. If
            `value` is \\(x\\), then `derivative` is
            \\(\\frac{f^\\prime(x)}{f(x)}\\), where \\(f\\) is the function
            representation of `LoggedTransform.transform`
        """
        return self.transform.derivative(value) / self.transform(value)
    
    def second_derivative(self, value):
        """
        Computes the second derivative of the function underlying this
        `LoggedTransform` at the given value(s).
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the derivative
        
        Returns
        -------
        derivative : number or sequence
            value of second derivative of transformation in same format as
            `value`. If `value` is \\(x\\), then `derivative` is
            \\(\\frac{f^{\\prime\\prime}(x)\\times f(x)-\
            \\big(f^\\prime(x)\\big)^2}{\\big(f(x)\\big)^2}\\)
        """
        func = self.transform(value)
        func_deriv = self.transform.derivative(value)
        func_deriv2 = self.transform.second_derivative(value)
        return ((func_deriv2 * func) - (func_deriv ** 2)) / (func ** 2)
    
    def third_derivative(self, value):
        """
        Computes the third derivative of the function underlying this
        `LoggedTransform` at the given value(s).
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the derivative
        
        Returns
        -------
        derivative : number or sequence
            value of third derivative of transformation in same format as
            `value`. If `value` is \\(x\\), then `derivative` is
            \\(\\frac{f^{\\prime\\prime\\prime}(x)\\times\\big(f(x)\\big)^2 -\
            3f^{\\prime\\prime}(x)\\times f^\\prime(x)\\times f(x) +\
            2\\big(f^\\prime(x)\\big)^3}{\\big(f(x)\\big)^3}\\), where \\(f\\)
            is the function representation of `LoggedTransform.transform`
        """
        func = self.transform(value)
        func_deriv = self.transform.derivative(value)
        func_deriv2 = self.transform.second_derivative(value)
        func_deriv3 = self.transform.third_derivative(value)
        return ((func_deriv3 * (func ** 2)) -\
            (3 * (func * func_deriv * func_deriv2)) +\
            (2 * (func_deriv ** 3))) / (func ** 3)
    
    def log_derivative(self, value):
        """
        Computes the natural logarithm of the absolute value of the derivative
        of the function underlying this `LoggedTransform` at the given
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
            \\(\\ln{\\big|f^\\prime(x)\\big|} - \\ln{\\big(f(x)\\big)}\\),
            where \\(f\\) is the function representation of
            `LoggedTransform.transform`
        """
        return self.transform.log_derivative(value) - self.apply(value)
    
    def derivative_of_log_derivative(self, value):
        """
        Computes the derivative of the natural logarithm of the absolute value
        of the derivative of the function underlying this `LoggedTransform` at
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
            then `derivative` is
            \\(\\frac{f^{\\prime\\prime}(x)}{f^\\prime(x)} -\
            \\frac{f^\\prime(x)}{f(x)}\\), where \\(f\\) is the function
            representation of `LoggedTransform.transform`
        """
        return self.transform.derivative_of_log_derivative(value) -\
            self.derivative(value)
    
    def second_derivative_of_log_derivative(self, value):
        """
        Computes the second derivative of the natural logarithm of the absolute
        value of the derivative of the function underlying this
        `LoggedTransform` at the given value(s).
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the derivative
        
        Returns
        -------
        derivative : number or sequence
            value of the second derivative of the log of the derivative of
            transformation in same format as `value`. If `value` is \\(x\\),
            then `derivative` is \\(\
            \\left(\\frac{f^{\\prime\\prime\\prime}(x)\\times f^\\prime(x) -\
            \\big(f^{\\prime\\prime}(x)\\big)^2}{\\big(f^\\prime(x)\\big)^2}\
            \\right) - \\left(\\frac{f^{\\prime\\prime}(x)\\times f(x) -\
            \\big(f^\\prime(x)\\big)^2}{\\big(f(x)\\big)^2}\\right)\\), where
            \\(f\\) is the function representation of
            `LoggedTransform.transform`
        """
        return self.transform.second_derivative_of_log_derivative(value) -\
            self.second_derivative(value)
    
    def apply(self, value):
        """
        Applies this `LoggedTransform` to the value and returns the result.
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the
            transformation
        
        Returns
        -------
        transformed : number or sequence
            transformed value same format as `value`. If `value` is \\(x\\),
            then `transformed` is \\(\\ln{\\big(f(x)\\big)}\\), where \\(f\\)
            is the function representation of `LoggedTransform.transform`
        """
        return np.log(np.abs(self.transform(value)))
    
    def apply_inverse(self, value):
        """
        Applies the inverse of this `LoggedTransform` to the value and returns
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
            then `inverted` is \\(f^{-1}\\big(e^y\\big)\\), where \\(f\\) is
            the function representation of `LoggedTransform.transform`
        """
        return self.transform.apply_inverse(np.exp(value))
    
    def to_string(self):
        """
        Generates a string version of this `LoggedTransform`.
        
        Returns
        -------
        representation : str
            `'ln[t]'`, where `t` is the string representation of
            `LoggedTransform.transform`
        """
        return 'ln[{!s}]'.format(self.transform.to_string())
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data about this `LoggedTransform`
        so it can be loaded later.
        
        Parameters
        ----------
        group : h5py.Group
            hdf5 file group to which to write data about this `LoggedTransform`
        """
        group.attrs['class'] = 'LoggedTransform'
        self.transform.fill_hdf5_group(group.create_group('transform'))
    
    def __eq__(self, other):
        """
        Checks the given object for equality with this `LoggedTransform`.
        
        Parameters
        ----------
        other : object
            object to check for equality
        
        Returns
        -------
        result : bool
            True if and only if `other` is another `LoggedTransform` with the
            same `LoggedTransform.transform`
        """
        if isinstance(other, LoggedTransform):
            return self.transform == other.transform
        else:
            return False

