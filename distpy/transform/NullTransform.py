"""
Module containing class representing a transformation of the form:
$$x\\longrightarrow x$$

**File**: $DISTPY/distpy/transform/NullTransform.py  
**Author**: Keith Tauscher  
**Date**: 17 May 2021
"""
from .Transform import Transform

class NullTransform(Transform):
    """
    Class representing a transformation of the form: $$x\\longrightarrow x$$
    """
    def derivative(self, value):
        """
        Computes the derivative of the function underlying this `NullTransform`
        at the given value(s).
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the derivative
        
        Returns
        -------
        derivative : number or sequence
            value of derivative of transformation in same format as `value`. If
            `value` is \\(x\\), then `derivative` is \\(1\\)
        """
        return (0. * value) + 1.
    
    def second_derivative(self, value):
        """
        Computes the second derivative of the function underlying this
        `NullTransform` at the given value(s).
        
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
        return 0. * value
    
    def third_derivative(self, value):
        """
        Computes the third derivative of the function underlying this
        `NullTransform` at the given value(s).
        
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
        return 0. * value
    
    def log_derivative(self, value):
        """
        Computes the natural logarithm of the absolute value of the derivative
        of the function underlying this `NullTransform` at the given value(s).
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the derivative
        
        Returns
        -------
        derivative : number or sequence
            value of the log of the derivative of transformation in same format
            as `value`. If `value` is \\(x\\), then `derivative` is \\(0\\)
        """
        return 0. * value
    
    def derivative_of_log_derivative(self, value):
        """
        Computes the derivative of the natural logarithm of the absolute value
        of the derivative of the function underlying this `NullTransform` at
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
        return 0. * value
    
    def second_derivative_of_log_derivative(self, value):
        """
        Computes the second derivative of the natural logarithm of the absolute
        value of the derivative of the function underlying this `NullTransform`
        at the given value(s).
        
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
        Applies this `NullTransform` to the value and returns the result.
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the
            transformation
        
        Returns
        -------
        transformed : number or sequence
            transformed value same format as `value`. If `value` is \\(x\\),
            then `transformed` is \\(x\\)
        """
        return 1 * value
    
    def apply_inverse(self, value):
        """
        Applies the inverse of this `NullTransform` to the value and returns
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
            then `inverted` is \\(y\\)
        """
        return 1 * value
    
    def __eq__(self, other):
        """
        Checks the given object for equality with this `NullTransform`.
        
        Parameters
        ----------
        other : object
            object to check for equality
        
        Returns
        -------
        result : bool
            True if and only if `other` is another `NullTransform`
        """
        return isinstance(other, NullTransform)
    
    def to_string(self):
        """
        Generates a string version of this `NullTransform`.
        
        Returns
        -------
        representation : str
            `'none'`
        """
        return 'none'
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data about this `NullTransform`
        so it can be loaded later.
        
        Parameters
        ----------
        group : h5py.Group
            hdf5 file group to which to write data about this `NullTransform`
        """
        group.attrs['class'] = 'NullTransform'

