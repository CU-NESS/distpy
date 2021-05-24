"""
Module containing class representing a transformation of the form:
$$x\\longrightarrow e^{f(x)}$$

**File**: $DISTPY/distpy/transform/ExponentiatedTransform.py  
**Author**: Keith Tauscher  
**Date**: 17 May 2021
"""
import numpy as np
from .Transform import Transform

class ExponentiatedTransform(Transform):
    """
    Class representing a transformation of the form:
    $$x\\longrightarrow e^{f(x)}$$
    """
    def __init__(self, transform):
        """
        Initializes a new `ExponentiatedTransform` which represents the
        following transformation: $$x\\longrightarrow e^{f(x)}$$
        
        Parameters
        ----------
        transform : `distpy.transform.Transform.Transform`
            transformation applied before exponentiation, \\(f\\)
        """
        self.transform = transform
    
    @property
    def transform(self):
        """
        The transform which this is the exponential of, \\(f\\).
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
            the transformation to exponentiate
        """
        if isinstance(value, Transform):
            self._transform = value
        else:
            raise TypeError("transform was not a Transform object.")
    
    def derivative(self, value):
        """
        Computes the derivative of the function underlying this
        `ExponentiatedTransform` at the given value(s).
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the derivative
        
        Returns
        -------
        derivative : number or sequence
            value of derivative of transformation in same format as `value`. If
            `value` is \\(x\\), then `derivative` is
            \\(e^{f(x)}\\times f^\\prime(x)\\), where \\(f\\) is the function
            representation of `ExponentiatedTransform.transform`
        """
        e_to_func = np.exp(self.transform(value))
        func_deriv = self.transform.derivative(value)
        return func_deriv * e_to_func
    
    def second_derivative(self, value):
        """
        Computes the second derivative of the function underlying this
        `ExponentiatedTransform` at the given value(s).
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the derivative
        
        Returns
        -------
        derivative : number or sequence
            value of second derivative of transformation in same format as
            `value`. If `value` is \\(x\\), then `derivative` is
            \\(\\big[\\big(f^\\prime(x)\\big)^2 + f^{\\prime\\prime}(x)\\big]\
            \\times e^{f(x)}\\), where \\(f\\) is the function representation
            of `ExponentiatedTransform.transform`
        """
        e_to_func = np.exp(self.transform(value))
        func_deriv = self.transform.derivative(value)
        func_deriv2 = self.transform.second_derivative(value)
        return (func_deriv2 + (func_deriv ** 2)) * e_to_func
    
    def third_derivative(self, value):
        """
        Computes the third derivative of the function underlying this
        `ExponentiatedTransform` at the given value(s).
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the derivative
        
        Returns
        -------
        derivative : number or sequence
            value of third derivative of transformation in same format as
            `value`. If `value` is \\(x\\), then `derivative` is
            \\(\\big[\\big(f^\\prime(x)\\big)^3 +\
            3\\times f^{\\prime\\prime}(x)\\times f^\\prime(x) +\
            f^{\\prime\\prime\\prime}(x)\\big] e^{f(x)}\\), where \\(f\\) is
            the function representation of `ExponentiatedTransform.transform`
        """
        e_to_func = np.exp(self.transform(value))
        func_deriv = self.transform.derivative(value)
        func_deriv2 = self.transform.second_derivative(value)
        func_deriv3 = self.transform.third_derivative(value)
        return e_to_func * (func_deriv3 + (3 * (func_deriv * func_deriv2)) +\
            (func_deriv ** 3))
    
    def log_derivative(self, value):
        """
        Computes the natural logarithm of the absolute value of the derivative
        of the function underlying this `ExponentiatedTransform` at the given
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
            \\(\\ln{\\big|f^\\prime(x)\\big|} + f(x)\\), where \\(f\\) is the
            function representation of `ExponentiatedTransform.transform`
        """
        return self.transform.log_derivative(value) + self.transform(value)
    
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
            then `derivative` is
            \\(\\frac{f^{\\prime\\prime}(x)}{f^\\prime(x)} + f^\\prime(x)\\),
            where \\(f\\) is the function representation of
            `ExponentiatedTransform.transform`
        """
        return self.transform.derivative_of_log_derivative(value) +\
            self.transform.derivative(value)
    
    def second_derivative_of_log_derivative(self, value):
        """
        Computes the second derivative of the natural logarithm of the absolute
        value of the derivative of the function underlying this
        `ExponentiatedTransform` at the given value(s).
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the derivative
        
        Returns
        -------
        derivative : number or sequence
            value of the second derivative of the log of the derivative of
            transformation in same format as `value`. If `value` is \\(x\\),
            then `derivative` is \\(\\left(\\frac{\
            f^{\\prime\\prime\\prime}(x)\\times f^\\prime(x) -\
            \\big(f^{\\prime\\prime}(x)\\big)^2}{\
            \\big(f^\\prime(x)\\big)^2}\\right) + f^{\\prime\\prime}(x)\\),
            where \\(f\\) is the function representation of
            `ExponentiatedTransform.transform`
        """
        return self.transform.second_derivative_of_log_derivative(value) +\
            self.transform.second_derivative(value)
    
    def apply(self, value):
        """
        Applies this `ExponentiatedTransform` to the value and returns the
        result.
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the
            transformation
        
        Returns
        -------
        transformed : number or sequence
            transformed value same format as `value`. If `value` is \\(x\\),
            then `transformed` is \\(e^{f(x)}\\), where \\(f\\) is the function
            representation of `ExponentiatedTransform.transform`
        """
        return np.exp(self.transform(value))
    
    def apply_inverse(self, value):
        """
        Applies the inverse of this `ExponentiatedTransform` to the value and
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
            then `inverted` is \\(f^{-1}\\big(\\ln{(y)}\\big)\\), where \\(f\\)
            is the function representation of
            `ExponentiatedTransform.transform`
        """
        return self.transform.apply_inverse(np.log(value))
    
    def to_string(self):
        """
        Generates a string version of this `ExponentiatedTransform`.
        
        Returns
        -------
        representation : str
            `'e^t'`, where `t` is the string representation of
            `ExponentiatedTransform.transform`
        """
        return 'e^{!s}'.format(self.transform.to_string())
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data about this
        `ExponentiatedTransform` so it can be loaded later.
        
        Parameters
        ----------
        group : h5py.Group
            hdf5 file group to which to write data about this
            `ExponentiatedTransform`
        """
        group.attrs['class'] = 'ExponentiatedTransform'
        self.transform.fill_hdf5_group(group.create_group('transform'))
    
    def __eq__(self, other):
        """
        Checks the given object for equality with this
        `ExponentiatedTransform`.
        
        Parameters
        ----------
        other : object
            object to check for equality
        
        Returns
        -------
        result : bool
            True if and only if `other` is another `ExponentiatedTransform`
            with the same `ExponentiatedTransform.transform`
        """
        if isinstance(other, ExponentiatedTransform):
            return self.transform == other.transform
        else:
            return False

