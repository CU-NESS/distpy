"""
Module containing class representing a transformation of the form:
$$x\\longrightarrow f\\big(g(x)\\big)$$

**File**: $DISTPY/distpy/transform/CompositeTransform.py  
**Author**: Keith Tauscher  
**Date**: 17 May 2021
"""
import numpy as np
from .Transform import Transform
from .NullTransform import NullTransform

class CompositeTransform(Transform):
    """
    Class representing a transformation of the form:
    $$x\\longrightarrow f\\big(g(x)\\big)$$
    """
    def __init__(self, inner_transform, outer_transform):
        """
        Initializes a new `CompositeTransform` which represents the following
        transformation: $$x\\longrightarrow f\\big(g(x)\\big)$$
        
        Parameters
        ----------
        inner_transform : `distpy.transform.Transform.Transform`
            first transformation applied, \\(f\\)
        translation : `distpy.transform.Transform.Transform`
            second transformation applied, \\(g\\)
        """
        self.inner_transform = inner_transform
        self.outer_transform = outer_transform
    
    @staticmethod
    def generate_from_list(*transforms):
        """
        Generates a `CompositeTransform` from an arbitrary number of
        `distpy.transform.Transform.Transform` objects. Since the
        `CompositeTransform` only works with two
        `distpy.transform.Transform.Transform` objects at a time, this static
        method will create a nested series of `CompositeTransform` and return
        the complete one.
        
        Parameters
        ----------
        transforms : sequence
            arbitrary length sequence of `distpy.transform.Transform.Transform`
            objects. They should be ordered in the same order as they are
            evaluated in the desired `CompositeTransform`
        
        Returns
        -------
        composite_transform : `CompositeTransform`
            object representing the composite of all given transformations
        """
        if len(transforms) == 0:
            return NullTransform()
        elif len(transforms) == 1:
            if isinstance(transforms[0], Transform):
                return transforms[0]
            else:
                raise TypeError("A non-Transform object was passed to the " +\
                    "generate_from_list function.")
        else:
            current_transform = transforms[0]
            for transform in transforms[1:]:
                current_transform =\
                    CompositeTransform(current_transform, transform)
            return current_transform
    
    @property
    def inner_transform(self):
        """
        The innermost (first applied) transform.
        """
        if not hasattr(self, '_inner_transform'):
            raise AttributeError("inner_transform referenced before it was " +\
                "set.")
        return self._inner_transform
    
    @inner_transform.setter
    def inner_transform(self, value):
        """
        Setter for `CompositeTransform.inner_transform`.
        
        Parameters
        ----------
        value : `distpy.transform.Transform.Transform`
            transformation to apply first
        """
        if isinstance(value, Transform):
            self._inner_transform = value
        else:
            raise TypeError("inner_transform given was not a Transform " +\
                "object.")
    
    @property
    def outer_transform(self):
        """
        The outermost (last applied) transform.
        """
        if not hasattr(self, '_outer_transform'):
            raise AttributeError("outer_transform referenced before it was " +\
                "set.")
        return self._outer_transform
    
    @outer_transform.setter
    def outer_transform(self, value):
        """
        Setter for `CompositeTransform.outer_transform`.
        
        Parameters
        ----------
        value : `distpy.transform.Transform.Transform`
            transformation to apply last
        """
        if isinstance(value, Transform):
            self._outer_transform = value
        else:
            raise TypeError("outer_transform given was not a Transform " +\
                "object.")
    
    def derivative(self, value):
        """
        Computes the derivative of the function underlying this
        `CompositeTransform` at the given value(s).
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the derivative
        
        Returns
        -------
        derivative : number or sequence
            value of derivative of transformation in same format as `value`. If
            `value` is \\(x\\), then `derivative` is
            \\(f^\\prime\\big(g(x)\\big)\\times g^\\prime(x)\\), where \\(f\\)
            and \\(g\\) are the function representations of
            `CompositeTransform.outer_transform` and
            `CompositeTransform.inner_transform`, respectively
        """
        inner_transformed_value = self.inner_transform(value)
        outer_derivative =\
            self.outer_transform.derivative(inner_transformed_value)
        inner_derivative = self.inner_transform.derivative(value)
        return outer_derivative * inner_derivative
    
    def second_derivative(self, value):
        """
        Computes the second derivative of the function underlying this
        `CompositeTransform` at the given value(s).
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the derivative
        
        Returns
        -------
        derivative : number or sequence
            value of second derivative of transformation in same format as
            `value`. If `value` is \\(x\\), then `derivative` is
            \\(f^{\\prime\\prime}\\big(g(x)\\big)\\times\
            \\big(g^\\prime(x)\\big)^2 + f^\\prime\\big(g(x)\\big)\\times\
            g^{\\prime\\prime}(x)\\)
        """
        inner_transformed_value = self.inner_transform(value)
        outer_first_derivative =\
            self.outer_transform.derivative(inner_transformed_value)
        outer_second_derivative =\
            self.outer_transform.second_derivative(inner_transformed_value)
        inner_first_derivative = self.inner_transform.derivative(value)
        inner_second_derivative = self.inner_transform.second_derivative(value)
        first_term = (outer_second_derivative * (inner_first_derivative ** 2))
        second_term = (outer_first_derivative * inner_second_derivative)
        return (first_term + second_term)
    
    def third_derivative(self, value):
        """
        Computes the third derivative of the function underlying this
        `CompositeTransform` at the given value(s).
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the derivative
        
        Returns
        -------
        derivative : number or sequence
            value of third derivative of transformation in same format as
            `value`. If `value` is \\(x\\), then `derivative` is
            \\(f^{\\prime\\prime\\prime}\\big(g(x)\\big)\\times\
            \\big(g^\\prime(x)\\big)^3 +\
            3\\times f^{\\prime\\prime}\\big(g(x)\\big)\\times\
            g^\\prime(x)\\times g^{\\prime\\prime}(x) +\
            f^\\prime\\big(g(x)\\big)\\times g^{\\prime\\prime\\prime}(x)\\),
            where \\(f\\) and \\(g\\) are the function representations of
            `CompositeTransform.outer_transform` and
            `CompositeTransform.inner_transform`, respectively
        """
        inner_transformed_value = self.inner_transform(value)
        outer_first_derivative =\
            self.outer_transform.derivative(inner_transformed_value)
        outer_second_derivative =\
            self.outer_transform.second_derivative(inner_transformed_value)
        outer_third_derivative =\
            self.outer_transform.third_derivative(inner_transformed_value)
        inner_first_derivative = self.inner_transform.derivative(value)
        inner_second_derivative = self.inner_transform.second_derivative(value)
        inner_third_derivative = self.inner_transform.third_derivative(value)
        first_term = (outer_third_derivative * (inner_first_derivative ** 3))
        second_term = (3 * (outer_second_derivative *\
            (inner_first_derivative * inner_second_derivative)))
        third_term = outer_first_derivative * inner_third_derivative
        return ((first_term + second_term) + third_term)
    
    def log_derivative(self, value):
        """
        Computes the natural logarithm of the absolute value of the derivative
        of the function underlying this `CompositeTransform` at the given
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
            \\(\\ln{\\big|f^\\prime\\big(g(x)\\big)\\big|}+\
            \\ln{\\big|g^\\prime(x)\\big|}\\), where \\(f\\) and \\(g\\) are
            the function representations of
            `CompositeTransform.outer_transform` and
            `CompositeTransform.inner_transform`, respectively
        """
        return np.log(np.abs(self.derivative(value)))
    
    def derivative_of_log_derivative(self, value):
        """
        Computes the derivative of the natural logarithm of the absolute value
        of the derivative of the function underlying this `CompositeTransform` 
        at the given value(s).
        
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
            \\(\\frac{f^{\\prime\\prime}\\big(g(x)\\big)\\times\
            g^\\prime(x)}{f^\\prime\\big(g(x)\\big)} +\
            \\frac{g^{\\prime\\prime}(x)}{g^\\prime(x)}\\), where \\(f\\) and
            \\(g\\) are the function representations of
            `CompositeTransform.outer_transform` and
            `CompositeTransform.inner_transform`, respectively
        """
        return self.second_derivative(value) / self.derivative(value)
    
    def second_derivative_of_log_derivative(self, value):
        """
        Computes the second derivative of the natural logarithm of the absolute
        value of the derivative of the function underlying this
        `CompositeTransform` at the given value(s).
        
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
        first = self.derivative(value)
        second = self.second_derivative(value)
        third = self.third_derivative(value)
        return (((first * third) - (second ** 2)) / (first ** 2))
    
    def apply(self, value):
        """
        Applies this `CompositeTransform` to the value and returns the result.
        
        Parameters
        ----------
        value : number or sequence
            number or sequence of numbers at which to evaluate the
            transformation
        
        Returns
        -------
        transformed : number or sequence
            transformed value same format as `value`. If `value` is \\(x\\),
            then `transformed` is \\(f\\big(g(x)\\big)\\), where \\(f\\) and
            \\(g\\) represent the functional forms of
            `CompositeTransform.outer_transform` and
            `CompositeTransform.inner_transform`, respectively
        """
        return self.outer_transform(self.inner_transform(value))
    
    def apply_inverse(self, value):
        """
        Applies the inverse of this `CompositeTransform` to the value and
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
            then `inverted` is \\(g^{-1}\\big(f^{-1}(y)\\big)\\), where \\(f\\)
            and \\(g\\) represent the functional forms of
            `CompositeTransform.outer_transform` and
            `CompositeTransform.inner_transform`, respectively
        """
        return self.inner_transform.I(self.outer_transform.I(value))
    
    def to_string(self):
        """
        Generates a string version of this `CompositeTransform`.
        
        Returns
        -------
        representation : str
            `'(o then i)'`, where `o` and `i` are the string representations of
            `CompositeTransform.outer_transform` and
            `CompositeTransform.inner_transform`, respectively
        """
        return '({0!s} then {1!s})'.format(self.inner_transform.to_string,\
            self.outer_transform.to_string)
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data about this
        `CompositeTransform` so it can be loaded later.
        
        Parameters
        ----------
        group : h5py.Group
            hdf5 file group to which to write data about this
            `CompositeTransform`
        """
        group.attrs['class'] = 'CompositeTransform'
        self.inner_transform.fill_hdf5_group(\
            group.create_group('inner_transform'))
        self.outer_transform.fill_hdf5_group(\
            group.create_group('outer_transform'))
    
    def __eq__(self, other):
        """
        Checks the given object for equality with this `CompositeTransform`.
        
        Parameters
        ----------
        other : object
            object to check for equality
        
        Returns
        -------
        result : bool
            True if and only if `other` is another `CompositeTransform` with
            the same `CompositeTransform.outer_transform` and
            `CompositeTransform.inner_transform`
        """
        if isinstance(other, CompositeTransform):
            inner_transforms_equal =\
                (self.inner_transform == other.inner_transform)
            outer_transforms_equal =\
                (self.outer_transform == other.outer_transform)
            return inner_transforms_equal and outer_transforms_equal
        else:
            return False

