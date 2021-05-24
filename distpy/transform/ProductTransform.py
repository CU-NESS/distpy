"""
Module containing class representing a transformation of the form:
$$x\\longrightarrow f(x)\\times g(x)$$

**File**: $DISTPY/distpy/transform/ProductTransform.py  
**Author**: Keith Tauscher  
**Date**: 17 May 2021
"""
import numpy as np
from ..util import sequence_types
from .Transform import Transform

class ProductTransform(Transform):
    """
    Class representing a transformation of the form:
    $$x\\longrightarrow f(x)\\times g(x)$$
    """
    def __init__(self, first_transform, second_transform):
        """
        Initializes a new `ProductTransform` which represents the following
        transformation: $$x\\longrightarrow f(x)\\times g(x)$$
        
        Parameters
        ----------
        first_transform : `distpy.transform.Transform.Transform`
            one of the transformations to multiply, \\(f\\)
        second_transform : `distpy.transform.Transform.Transform`
            the other of the transformations to multiply, \\(g\\)
        """
        self.transforms = [first_transform, second_transform]
    
    @property
    def transforms(self):
        """
        Length-2 list of `distpy.transform.Transform.Transform` objects being
        multiplied, \\(f\\) and \\(g\\).
        """
        if not hasattr(self, '_transforms'):
            raise AttributeError("transforms referenced before it was set.")
        return self._transforms
    
    @transforms.setter
    def transforms(self, value):
        """
        Setter for `ProductTransform.transforms`.
        
        Parameters
        ----------
        value : sequence
            must be a length-2 list of `distpy.transform.Transform.Transform`
            objects
        """
        if type(value) in sequence_types:
            if len(value) == 2:
                if all([isinstance(element, Transform) for element in value]):
                    self._transforms = value
                else:
                    raise TypeError("At least one element of the " +\
                        "transforms sequence was not a Transform object.")
            else:
                raise ValueError("More than two transforms were given.")
        else:
            raise TypeError("transforms was set to a non-sequence.")
    
    def derivative(self, value):
        """
        Computes the derivative of the function underlying this
        `ProductTransform` at the given value(s).
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the derivative
        
        Returns
        -------
        derivative : number or sequence
            value of derivative of transformation in same format as `value`. If
            `value` is \\(x\\), then `derivative` is
            \\(f^\\prime(x)\\times g(x) + f(x)\\times g^\\prime(x)\\), where
            \\(f\\) and \\(g\\) are the function representations of the
            elements of `ProductTransform.transforms`
        """
        funcs = [transform(value) for transform in self.transforms]
        func_derivs =\
            [transform.derivative(value) for transform in self.transforms]
        return ((funcs[0] * func_derivs[1]) + (funcs[1] * func_derivs[0]))
    
    def second_derivative(self, value):
        """
        Computes the second derivative of the function underlying this
        `ProductTransform` at the given value(s).
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the derivative
        
        Returns
        -------
        derivative : number or sequence
            value of second derivative of transformation in same format as
            `value`. If `value` is \\(x\\), then `derivative` is
            \\(f^{\\prime\\prime}(x)\\times g(x) +\
            2\\times f^\\prime(x)\\times g^\\prime(x) +\
            f(x)\\times g^{\\prime\\prime}(x)\\), where \\(f\\) and \\(g\\) are
            the function representations of the elements of
            `ProductTransform.transforms`
        """
        funcs = [transform(value) for transform in self.transforms]
        func_derivs =\
            [transform.derivative(value) for transform in self.transforms]
        func_derivs2 = [transform.second_derivative(value)\
            for transform in self.transforms]
        return ((funcs[0] * func_derivs2[1]) + (funcs[1] * func_derivs2[0]) +\
            (2 * (func_derivs[0] * func_derivs[1])))
    
    def third_derivative(self, value):
        """
        Computes the third derivative of the function underlying this
        `ProductTransform` at the given value(s).
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the derivative
        
        Returns
        -------
        derivative : number or sequence
            value of third derivative of transformation in same format as
            `value`. If `value` is \\(x\\), then `derivative` is
            \\(f^{\\prime\\prime\\prime}(x)\\times g(x) +\
            3\\times f^{\\prime\\prime}(x)\\times g^\\prime(x) +\
            3\\times f^\\prime(x)\\times g^{\\prime\\prime}(x) +\
            f(x)\\times g^{\\prime\\prime\\prime}(x)\\), where \\(f\\) and
            \\(g\\) are the function representations of the elements of
            `ProductTransform.transforms`
        """
        funcs = [transform(value) for transform in self.transforms]
        func_derivs =\
            [transform.derivative(value) for transform in self.transforms]
        func_derivs2 = [transform.second_derivative(value)\
            for transform in self.transforms]
        func_derivs3 = [transform.third_derivative(value)\
            for transform in self.transforms]
        return ((funcs[0] * func_derivs3[1]) + (funcs[1] * func_derivs3[0]) +\
            (3 * ((func_derivs[0] * func_derivs2[1]) +\
            (func_derivs[1] * func_derivs2[0]))))
    
    def log_derivative(self, value):
        """
        Computes the natural logarithm of the absolute value of the derivative
        of the function underlying this `ProductTransform` at the given
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
            \\(\\ln{\\big|f^\\prime(x)\\times g(x) +\
            f(x)\\times g^\\prime(x)\\big|}\\), where \\(f\\) and \\(g\\) are
            the function representations of the elements of
            `ProductTransform.transforms`
        """
        return np.log(np.abs(self.derivative(value)))
    
    def derivative_of_log_derivative(self, value):
        """
        Computes the derivative of the natural logarithm of the absolute value
        of the derivative of the function underlying this `ProductTransform` at
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
            then `derivative` is \\(\\frac{f^{\\prime\\prime}(x)\\times g(x) +\
            2\\times f^\\prime(x)\\times g^\\prime(x) +\
            f(x)\\times g^{\\prime\\prime}(x)}{f^\\prime(x)\\times g(x) +\
            f(x) \\times g^\\prime(x)}\\), where \\(f\\) and \\(g\\) are the
            function representations of the elements of
            `ProductTransform.transforms`
        """
        return self.second_derivative(value) / self.derivative(value)
    
    def second_derivative_of_log_derivative(self, value):
        """
        Computes the second derivative of the natural logarithm of the absolute
        value of the derivative of the function underlying this
        `ProductTransform` at the given value(s).
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the derivative
        
        Returns
        -------
        derivative : number or sequence
            value of the second derivative of the log of the derivative of
            transformation in same format as `value`.
        """
        func_deriv = self.derivative(value)
        func_deriv2 = self.second_derivative(value)
        func_deriv3 = self.third_derivative(value)
        return (((func_deriv * func_deriv3) - (func_deriv2 ** 2)) /\
            (func_deriv ** 2))
    
    def apply(self, value):
        """
        Applies this `ProductTransform` to the value and returns the result.
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the
            transformation
        
        Returns
        -------
        transformed : number or sequence
            transformed value same format as `value`. If `value` is \\(x\\),
            then `transformed` is \\(f(x)\\times g(x)\\), where \\(f\\) and
            \\(g\\) are the function representations of the elements of
            `ProductTransform.transforms`
        """
        return self.transforms[0](value) * self.transforms[1](value)
    
    def apply_inverse(self, value):
        """
        This method raises a `NotImplementedError` because general
        `ProductTransform` objects are not invertible.
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the inverse
            transformation
        """
        raise NotImplementedError("The ProductTransform cannot be inverted.")
    
    def to_string(self):
        """
        Generates a string version of this `ProductTransform`.
        
        Returns
        -------
        representation : str
            `'(f*s)'`, where `f` and `s` are the string representations of the
            elements of `ProductTransform.transforms`
        """
        return '({0!s}*{1!s})'.format(self.transforms[0].to_string(),\
            self.transforms[1].to_string())
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data about this `ProductTransform`
        so it can be loaded later.
        
        Parameters
        ----------
        group : h5py.Group
            hdf5 file group to which to write data about this
            `ProductTransform`
        """
        group.attrs['class'] = 'ProductTransform'
        self.transforms[0].fill_hdf5_group(group.create_group('transform_0'))
        self.transforms[1].fill_hdf5_group(group.create_group('transform_1'))
    
    def __eq__(self, other):
        """
        Checks the given object for equality with this `ProductTransform`.
        
        Parameters
        ----------
        other : object
            object to check for equality
        
        Returns
        -------
        result : bool
            True if and only if `other` is another `ProductTransform` with the
            same `ProductTransform.transforms`
        """
        if isinstance(other, ProductTransform):
            transforms_same = ((self.transforms[0] == other.transforms[0]) and\
                (self.transforms[1] == other.transforms[1]))
            if transforms_same:
                return True
            transforms_flipped =\
                ((self.transforms[0] == other.transforms[1]) and\
                (self.transforms[1] == other.transforms[0]))
            if transforms_flipped:
                return True
        return False

