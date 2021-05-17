"""
Module containing base class for all transformations.

**File**: $DISTPY/distpy/transform/Transform.py  
**Author**: Keith Tauscher  
**Date**: 17 May 2021
"""
import numpy as np
from ..util import Savable

cannot_instantiate_transform_error =\
    NotImplementedError("Transform cannot be directly instantiated.")

class Transform(Savable):
    """
    Base class for all transformations.
    """
    def derivative(self, value):
        """
        Computes the derivative of the function underlying this `Transform` at
        the given value(s). This method must be implemented by all subclasses
        of `Transform`.
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the derivative
        
        Returns
        -------
        derivative : number or sequence
            value of derivative of transformation in same format as `value`
        """
        raise cannot_instantiate_transform_error
    
    def second_derivative(self, value):
        """
        Computes the second derivative of the function underlying this
        `Transform` at the given value(s). This method must be implemented by
        all subclasses  of `Transform`.
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the second
            derivative
        
        Returns
        -------
        derivative : number or sequence
            value of second derivative of transformation in same format as
            `value`
        """
        raise cannot_instantiate_transform_error
    
    def third_derivative(self, value):
        """
        Computes the third derivative of the function underlying this
        `Transform` at the given value(s). This method must be implemented by
        all subclasses  of `Transform`.
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the third
            derivative
        
        Returns
        -------
        derivative : number or sequence
            value of third derivative of transformation in same format as
            `value`
        """
        raise cannot_instantiate_transform_error
    
    def log_derivative(self, value):
        """
        Computes the natural logarithm of the absolute value of the derivative
        of the function underlying this `Transform` at the given value(s). This
        method must be implemented by all subclasses of `Transform`.
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the log
            derivative
        
        Returns
        -------
        derivative : number or sequence
            value of log derivative of transformation in same format as `value`
        """
        raise cannot_instantiate_transform_error
    
    def derivative_of_log_derivative(self, value):
        """
        Computes the derivative of the natural logarithm of the derivative of
        the function underlying this `Transform` at the given value(s). This
        method must be implemented by all subclasses of `Transform`.
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the derivative
            of the log derivative
        
        Returns
        -------
        derivative : number or sequence
            value of derivative of log derivative of transformation in same
            format as `value`
        """
        raise cannot_instantiate_transform_error
    
    def second_derivative_of_log_derivative(self, value):
        """
        Computes the second derivative of the natural logarithm of the
        derivative of the function underlying this `Transform` at the given
        value(s). This method must be implemented by all subclasses of
        `Transform`.
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the second
            derivative of the log derivative
        
        Returns
        -------
        derivative : number or sequence
            value of second derivative of log derivative of transformation in
            same format as `value`
        """
        raise cannot_instantiate_transform_error
    
    def apply(self, value):
        """
        Applies this `Transform` to the value and returns the result. This
        method must be implemented by all subclasses of `Transform`.
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the
            transformation
        
        Returns
        -------
        transformed : number or sequence
            transformed value same format as `value`
        """
        raise cannot_instantiate_transform_error
    
    def __call__(self, value):
        """
        Applies this `Transform` to the value and returns the result.
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the
            transformation
        
        Returns
        -------
        transformed : number or sequence
            transformed value same format as `value`
        """
        return self.apply(value)
    
    def apply_inverse(self, value):
        """
        Applies the inverse of this `Transform` to the value and returns the
        result. This method must be implemented by all subclasses of
        `Transform`.
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the inverse
            transformation
        
        Returns
        -------
        inverted : number or sequence
            untransformed value same format as `value`
        """
        raise cannot_instantiate_transform_error
    
    def I(self, value):
        """
        Applies the inverse of this `Transform` to the value and returns the
        result (alias for `Transform.apply_inverse`).
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the inverse
            transformation
        
        Returns
        -------
        inverted : number or sequence
            untransformed value same format as `value`
        """
        return self.apply_inverse(value)
    
    def untransform_minimum(self, transformed_minimum):
        """
        Untransforms the given minimum.
        
        Parameters
        ----------
        transformed_minimum : number or None
            the minimum in transformed space with None representing -np.inf
        
        Returns
        -------
        untransformed_minimum : number or None
            if `untransformed_minimum` is finite, it is returned.  
            Otherwise, None is returned to indicate that
            `untransformed_minimum` is minus infinity
        """
        if type(transformed_minimum) is type(None):
            untransformed_minimum = self.apply_inverse(-np.inf)
        else:
            untransformed_minimum = self.apply_inverse(transformed_minimum)
        if np.isfinite(untransformed_minimum):
            return untransformed_minimum
        else:
            return None
    
    def untransform_maximum(self, transformed_maximum):
        """
        Untransforms the given maximum.
        
        Parameters
        ----------
        transformed_maximum : number or None
            the maximum in transformed space with None representing +np.inf
        
        Returns
        -------
        untransformed_maximum : number or None
            if `untransformed_maximum` is finite, it is returned.  
            Otherwise, None is returned to indicate that
            `untransformed_maximum` is plus infinity
        """
        if type(transformed_maximum) is type(None):
            untransformed_maximum = self.apply_inverse(np.inf)
        else:
            untransformed_maximum = self.apply_inverse(transformed_maximum)
        if np.isfinite(untransformed_maximum):
            return untransformed_maximum
        else:
            return None
    
    def transform_minimum(self, untransformed_minimum):
        """
        Transforms the given minimum.
        
        Parameters
        ----------
        untransformed_minimum : number or None
            the minimum in untransformed space with None representing -np.inf
        
        Returns
        -------
        transformed_minimum : number or None
            if `transformed_minimum` is finite, it is returned.  
            Otherwise, None is returned to indicate that
            `transformed_minimum` is minus infinity
        """
        if type(untransformed_minimum) is type(None):
            transformed_minimum = self.apply(-np.inf)
        else:
            transformed_minimum = self.apply(untransformed_minimum)
        if np.isfinite(transformed_minimum):
            return transformed_minimum
        else:
            return None
    
    def transform_maximum(self, untransformed_maximum):
        """
        Transforms the given maximum.
        
        Parameters
        ----------
        untransformed_maximum : number or None
            the maximum in untransformed space with None representing +np.inf
        
        Returns
        -------
        transformed_maximum : number or None
            if `transformed_maximum` is finite, it is returned.  
            Otherwise, None is returned to indicate that
            `transformed_maximum` is plus infinity
        """
        if type(untransformed_maximum) is type(None):
            transformed_maximum = self.apply(np.inf)
        else:
            transformed_maximum = self.apply(untransformed_maximum)
        if np.isfinite(transformed_maximum):
            return transformed_maximum
        else:
            return None
    
    def to_string(self):
        """
        Generates a string version of this `Transform`. This method must be
        implemented by all subclasses of `Transform`.
        
        Returns
        -------
        representation : str
            string that can be cast into this `Transform`
        """
        raise cannot_instantiate_transform_error
    
    def __str__(self):
        """
        Generates a string version of this `Transform`. It calls the subclass
        implementation of the `Transform.to_string` method.
        
        Returns
        -------
        representation : str
            string that can be cast into this `Transform`
        """
        return self.to_string()
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data about this transform. This
        method must be implemented by all subclasses of `Transform`.
        
        Parameters
        ----------
        group : h5py.Group
            hdf5 file group to which to write data about this `Transform`
        """
        raise cannot_instantiate_transform_error
    
    def __eq__(self, other):
        """
        Fills the given hdf5 file group with data about this transform. This
        method must be implemented by all subclasses of `Transform`.
        
        Parameters
        ----------
        other : object
            another object to check for equality
        
        Returns
        -------
        result : bool
            True if and only if both `self` and `other` are the same
        """
        raise cannot_instantiate_transform_error

    def __ne__(self, other):
        """
        Asserts that checks for equality are consistent with checks for
        inequality.
        
        Parameters
        ----------
        other : object
            another object to check for inequality
        
        Returns
        -------
        result : bool
            False if and only if both `self` and `other` are the same
        """
        return (not self.__eq__(other))
    
    def __bool__(self):
        """
        This method makes it so that if-statements can be performed with
        variables storing `Transform` objects as their expressions. If the
        variable contains a non-None `Transform`, it will return False.
        
        Returns
        -------
        result : bool
            True
        """
        return True

