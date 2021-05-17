"""
Module containing a class representing a list of
`distpy.transform.Transform.Transform` objects, which allows it represent
(separable) multivariate transformations. It has methods to transform (and
untransform) first derivative vectors (gradients) and second derivative
matrices (hessians).

**File**: $DISTPY/distpy/transform/TransformList.py  
**Author**: Keith Tauscher  
**Date**: 16 May 2021
"""
from __future__ import division
from ..util import int_types, sequence_types, Savable, Loadable
from .NullTransform import NullTransform
from .CompositeTransform import CompositeTransform
from .CastTransform import cast_to_transform, castable_to_transform
from .LoadTransform import load_transform_from_hdf5_group
from .InvertTransform import invert_transform
from .CastTransform import castable_to_transform, cast_to_transform

class TransformList(Savable, Loadable):
    """
    Class representing a list of `distpy.transform.Transform.Transform`
    objects, which allows it represent (separable) multivariate
    transformations. It has methods to transform (and untransform) first
    derivative vectors (gradients) and second derivative matrices (hessians).
    """
    def __init__(self, *transforms):
        """
        Initializes a new `TransformList`.
        
        Parameters
        ----------
        transforms : sequence
            `distpy.transform.Transform.Transform` objects or objects which can
            be cast to `distpy.transform.Transform.Transform` objects using the
            the `distpy.transform.CastTransform.cast_to_transform` function.
        """
        self.transforms = transforms
    
    @staticmethod
    def cast(key, num_transforms=None):
        """
        Casts an object into a `TransformList` object. If num_transforms is
        non-None, this function can also cast to a TransformList object of a
        specific length.
        
        Parameters
        ----------
        key : `TransformList` or `distpy.transform.Transform.Transform` or\
        sequence
            object to cast to a `TransformList`, either a `TransformList`, a
            sequence of `distpy.transform.Transform.Transform` objects (or
            things that can be cast to them using the
            `distpy.transform.CastTransform.cast_to_transform` function), or a
            single `distpy.transform.Transform.Transform` object
        num_transforms : int or None
            - if `num_transforms` is None:
                - if `key` is a `TransformList` or list of
                `distpy.transform.Transform.Transform` objects (or things that
                can be cast to them using the
                `distpy.transform.CastTransform.cast_to_transform` function),
                then `key` implies `num_transforms` without it needing to be
                given
                - if `key` is a `distpy.transform.Transform.Transform` (or
                something that can be cast to one using the
                `distpy.transform.CastTransform.cast_to_transform` function), a
                `TransformList of length 1 is returned containing only that
                `distpy.transform.Transform.Transform`
            - if `num_transforms` is a positive integer:
                - if `key` is a `TransformList` or sequence of
                `distpy.transform.Transform.Transform` objects (or things that
                can be cast to transforms using the
                `distpy.transform.CastTransform.cast_to_transform function),
                `key` is checked to ensure it has this length
                - if `key` is a `distpy.transform.Transform.Transform`, `key`
                is repeated this many times in the returned
                `TransformList`.
        
        Returns
        -------
        transform_list : `TransformList`
            object casted from the key, guaranteed to have length
            num_transforms if `num_transforms` is not None
        """
        if isinstance(key, TransformList):
            if (type(num_transforms) is not type(None)) and\
                (len(key) != num_transforms):
                raise ValueError("The given TransformList was not of the " +\
                    "specified length. So, it could not be cast " +\
                    "successfully into a TransformList of the desired size.")
            else:
                return key
        elif type(key) in sequence_types:
            if (type(num_transforms) is not type(None)) and\
                (len(key) != num_transforms):
                raise ValueError("The given sequence was not of the " +\
                    "specified length. So, it could not be cast " +\
                    "successfully into a TransformList of the desired size.")
            else:
                return TransformList(*key)
        elif castable_to_transform(key):
            transform = cast_to_transform(key)
            if type(num_transforms) is type(None):
                return TransformList(transform)
            else:
                return TransformList(*([transform] * num_transforms))
        else:
            raise TypeError("key could not be cast to a TransformList object.")
    
    @staticmethod
    def castable(key, num_transforms=None,\
        return_transform_list_if_true=False):
        """
        Function determining whether the given key can be cast into a
        `TransformList` object.
        
        Parameters
        ----------
        key : object
            object to attempt to check for castability (see
            `TransformList.cast` static method for what types of `key` will
            work)
        num_transforms : int or None
            number of transformations to store in casted object (see
            `TransformList.cast` static method for details on this parameter)
        return_transform_list_if_true : bool
            determines what should be returned if `key` can be successfully
            cast to a `TransformList` with the desired number of elements
        
        Returns
        -------
        cast_result : bool or `TransformList`
            - if `key` can be successfully cast to a
            `TransformList` with the desired number of elements, this method
            returns:
                - the casted `TransformList` if `return_transform_list_if_true`
                is True
                - True if `return_transform_list_if_true` is False
            - if `key` cannot be successfully cast to a `TransformList` with
            the desired number of elements, this method returns False
        """
        try:
            transform_list =\
                TransformList.cast(key, num_transforms=num_transforms)
        except:
            return False
        else:
            if return_transform_list_if_true:
                return transform_list
            else:
                return True
    
    @property
    def transforms(self):
        """
        List of `distpy.transform.Transform.Transform` objects at the heart of
        this container.
        """
        if not hasattr(self, '_transforms'):
            raise AttributeError("transforms referenced before it was set.")
        return self._transforms
    
    @transforms.setter
    def transforms(self, value):
        """
        Setter for the sequence of `distpy.transform.Transform.Transform`
        objects at the heart of this object.
        
        Parameters
        ----------
        value : sequence
            `distpy.transform.Transform.Transform` objects or objects which can
            be cast to `distpy.transform.Transform.Transform` objects using the
            the `distpy.transform.CastTransform.cast_to_transform` function.
        """
        if type(value) in sequence_types:
            if all([castable_to_transform(element) for element in value]):
                self._transforms =\
                    [cast_to_transform(element) for element in value]
            else:
                raise ValueError("Not all elements of the transforms " +\
                    "sequence could be cast to Transform objects.")
        else:
            raise TypeError("transforms was set to a non-sequence.")
    
    @property
    def num_transforms(self):
        """
        The number of `distpy.transform.Transform.Transform` objects in this
        `TransformList`.
        """
        if not hasattr(self, '_num_transforms'):
            self._num_transforms = len(self.transforms)
        return self._num_transforms
    
    def __len__(self):
        """
        Finds the length of this `TransformList`.
        
        Returns
        -------
        length : int
            number of `distpy.transform.Transform.Transform` objects in this
            `TransformList`
        """
        return self.num_transforms
    
    def apply(self, untransformed_point, axis=-1):
        """
        Transforms the given point(s) from the untransformed space to the
        transformed space.
        
        Parameters
        ----------
        untransformed_point : numpy.ndarray
            untransformed point values
        axis : int
            the index of the axis of the array corresponding to the parameters
            to transform
        
        Returns
        -------
        transformed_point : numpy.ndarray
            the parameter values in the transformed point space in a
            `numpy.ndarray` of same shape as `untransformed_point`
        """
        point = untransformed_point.copy()
        axis = (axis % point.ndim)
        buffer_slice = ((slice(None),) * axis)
        for (itransform, transform) in enumerate(self.transforms):
            this_slice = buffer_slice + (itransform,)
            point[this_slice] = transform(point[this_slice])
        return point
    
    def __call__(self, untransformed_point, axis=-1):
        """
        Transforms the given point(s) from the untransformed space to the
        transformed space. This method simply calls the
        `TransformList.__call__` method.
        
        Parameters
        ----------
        untransformed_point : numpy.ndarray
            untransformed point values
        axis : int
            the index of the axis of the array corresponding to the parameters
            to transform
        
        Returns
        -------
        transformed_point : numpy.ndarray
            the parameter values in the transformed point space in a
            `numpy.ndarray` of same shape as `untransformed_point`
        """
        return self.apply(untransformed_point, axis=axis)
    
    def derivative(self, untransformed_point, axis=-1):
        """
        Finds the derivatives of the underlying transformations at the given
        point(s).
        
        Parameters
        ----------
        untransformed_point : numpy.ndarray
            untransformed point values
        axis : int
            the index of the axis of the array corresponding to the parameters
            to transform
        
        Returns
        -------
        derivatives : numpy.ndarray
            the derivatives of the transformations in a `numpy.ndarray` of same
            shape as `untransformed_point`
        """
        point = untransformed_point.copy()
        axis = (axis % point.ndim)
        buffer_slice = ((slice(None),) * axis)
        for (itransform, transform) in enumerate(self.transforms):
            this_slice = buffer_slice + (itransform,)
            point[this_slice] = transform.derivative(point[this_slice])
        return point
    
    def second_derivative(self, untransformed_point, axis=-1):
        """
        Finds the second derivatives of the underlying transformations at the
        given point(s).
        
        Parameters
        ----------
        untransformed_point : numpy.ndarray
            untransformed point values
        axis : int
            the index of the axis of the array corresponding to the parameters
            to transform
        
        Returns
        -------
        derivatives : numpy.ndarray
            the second derivatives of the transformations in a `numpy.ndarray`
            of same shape as `untransformed_point`
        """
        point = untransformed_point.copy()
        axis = (axis % point.ndim)
        buffer_slice = ((slice(None),) * axis)
        for (itransform, transform) in enumerate(self.transforms):
            this_slice = buffer_slice + (itransform,)
            point[this_slice] = transform.second_derivative(point[this_slice])
        return point
    
    def third_derivative(self, untransformed_point, axis=-1):
        """
        Finds the third derivatives of the underlying transformations at the
        given point(s).
        
        Parameters
        ----------
        untransformed_point : numpy.ndarray
            untransformed point values
        axis : int
            the index of the axis of the array corresponding to the parameters
            to transform
        
        Returns
        -------
        derivatives : numpy.ndarray
            the third derivatives of the transformations in a `numpy.ndarray`
            of same shape as `untransformed_point`
        """
        point = untransformed_point.copy()
        axis = (axis % point.ndim)
        buffer_slice = ((slice(None),) * axis)
        for (itransform, transform) in enumerate(self.transforms):
            this_slice = buffer_slice + (itransform,)
            point[this_slice] = transform.third_derivative(point[this_slice])
        return point
    
    def log_derivative(self, untransformed_point, axis=-1):
        """
        Finds the natural log of the derivatives of the underlying
        transformations at the given point(s).
        
        Parameters
        ----------
        untransformed_point : numpy.ndarray
            untransformed point values
        axis : int
            the index of the axis of the array corresponding to the parameters
            to transform
        
        Returns
        -------
        derivatives : numpy.ndarray
            the log derivatives of the transformations in a `numpy.ndarray` of
            same shape as `untransformed_point`
        """
        point = untransformed_point.copy()
        axis = (axis % point.ndim)
        buffer_slice = ((slice(None),) * axis)
        for (itransform, transform) in enumerate(self.transforms):
            this_slice = buffer_slice + (itransform,)
            point[this_slice] = transform.log_derivative(point[this_slice])
        return point
    
    def derivative_of_log_derivative(self, untransformed_point, axis=-1):
        """
        Finds the derivatives of the log derivatives of the underlying
        transformations at the given point(s).
        
        Parameters
        ----------
        untransformed_point : numpy.ndarray
            untransformed point values
        axis : int
            the index of the axis of the array corresponding to the parameters
            to transform
        
        Returns
        -------
        derivatives : numpy.ndarray
            the derivatives of the log derivatives of the transformations in a
            `numpy.ndarray` of same shape as `untransformed_point`
        """
        point = untransformed_point.copy()
        axis = (axis % point.ndim)
        buffer_slice = ((slice(None),) * axis)
        for (itransform, transform) in enumerate(self.transforms):
            this_slice = buffer_slice + (itransform,)
            point[this_slice] =\
                transform.derivative_of_log_derivative(point[this_slice])
        return point
    
    def second_derivative_of_log_derivative(self, untransformed_point,\
        axis=-1):
        """
        Finds the second derivatives of the log derivatives of the underlying
        transformations at the given point(s).
        
        Parameters
        ----------
        untransformed_point : numpy.ndarray
            untransformed point values
        axis : int
            the index of the axis of the array corresponding to the parameters
            to transform
        
        Returns
        -------
        derivatives : numpy.ndarray
            the second derivatives of the log derivatives of the
            transformations in a `numpy.ndarray` of same shape as
            `untransformed_point`
        """
        point = untransformed_point.copy()
        axis = (axis % point.ndim)
        buffer_slice = ((slice(None),) * axis)
        for (itransform, transform) in enumerate(self.transforms):
            this_slice = buffer_slice + (itransform,)
            point[this_slice] = transform.second_derivative_of_log_derivative(\
                point[this_slice])
        return point
    
    def apply_inverse(self, transformed_point, axis=-1):
        """
        Detransforms the given point(s) from the transformed space to the
        untransformed space.
        
        Parameters
        ----------
        transformed_point : numpy.ndarray
            transformed point values
        axis : int
            the index of the axis of the array corresponding to the parameters
            to untransform
        
        Returns
        -------
        untransformed_point : numpy.ndarray
            untransformed parameter values in a `numpy.ndarray` of same shape
            as `transformed_point`
        """
        point = transformed_point.copy()
        axis = (axis % point.ndim)
        buffer_slice = ((slice(None),) * axis)
        for (itransform, transform) in enumerate(self.transforms):
            this_slice = buffer_slice + (itransform,)
            point[this_slice] = transform.apply_inverse(point[this_slice])
        return point
    
    def I(self, transformed_point, axis=-1):
        """
        Detransforms the given point(s) from the transformed space to the
        untransformed space. This method simply calls the
        `TransformList.apply_inverse` method.
        
        Parameters
        ----------
        transformed_point : numpy.ndarray
            transformed point values
        axis : int
            the index of the axis of the array corresponding to the parameters
            to untransform
        
        Returns
        -------
        untransformed_point : numpy.ndarray
            untransformed parameter values in a `numpy.ndarray` of same shape
            as `transformed_point`
        """
        return self.apply_inverse(transformed_point, axis=axis)
    
    def detransform_gradient(self, transformed_gradient, untransformed_point,\
        axis=-1):
        """
        Detransforms the gradient (assuming it was evaluated at the given
        point) from the transformed space to the untransformed space. Assuming
        \\(\\frac{\\partial f}{\\partial y_i}\\) is the gradient in the
        transformed space, \\(y\\), this function encodes the equality:
        $$\\frac{\\partial f}{\\partial x_i} = \
        \\left(\\frac{\\partial f}{\\partial y_i}\\right) \\times \
        \\left(\\frac{dy_i}{dx_i}\\right).$$
        
        Parameters
        ----------
        transformed_gradient : numpy.ndarray
            the gradient in the transformed space,
            \\(\\frac{\\partial f}{\\partial y_i}\\)
        untransformed_point : numpy.ndarray
            the input point in the untransformed space, \\(x\\)
        axis : int
            integer index of the axis of the derivative in
            `transformed_gradient`
        
        Returns
        -------
        untransformed_gradient : numpy.ndarray
            gradient in untransformed space,
            \\(\\frac{\\partial f}{\\partial x_i}\\), in a `numpy.ndarray` of
            same shape as `transformed_gradient`
        """
        gradient = transformed_gradient.copy()
        axis = (axis % gradient.ndim)
        buffer_slice = ((slice(None),) * axis)
        for (itransform, transform) in enumerate(self.transforms):
            gradient[buffer_slice + (itransform,)] *=\
                transform.derivative(untransformed_point[itransform])
        return gradient
    
    def transform_covariance(self, untransformed_covariance,\
        untransformed_point, axis=(-2, -1)):
        """
        Uses the `TransformList.detransform_gradient` method twice to change
        the given covariance matrix from untransformed space to transformed
        space. Mathematically, this performs $$C^\\prime_{ij} =\
        C_{ij}\\times\\left(\\frac{dy_i}{dx_i}\\right)\
        \\times \\left(\\frac{dy_j}{dx_j}\\right).$$
        
        Parameters
        ----------
        untransformed_covariance : numpy.ndarray
            the covariance in the untransformed space, \\(C_{ij}\\)
        untransformed_point : numpy.ndarray
            point at which the covariance is defined in the untransformed space
        axis : sequence
            a 2-tuple containing the two integer indices of the axes
            representing the covariance matrix.
        
        Returns
        -------
        transformed_covariance : numpy.ndarray
            the covariance in untransformed space, \\(C^\\prime_{ij}\\), in
            `numpy.ndarray` of same shape as `untransformed_covariance`
        """
        covariance = untransformed_covariance.copy()
        covariance = self.detransform_gradient(untransformed_covariance,\
            untransformed_point, axis=axis[0])
        covariance = self.detransform_gradient(untransformed_covariance,\
            untransformed_point, axis=axis[1])
        return covariance
    
    def transform_gradient(self, untransformed_gradient, untransformed_point,\
        axis=-1):
        """
        Transforms the gradient (assuming it was evaluated at the given
        point) from the untransformed space to the transformed space. Assuming
        that \\(\\frac{\\partial f}{\\partial x_i}\\) is the derivative in the
        untransformed space, \\(x\\), and the transformed space is \\(y\\),
        this function encodes the following equality:
        $$\\frac{\\partial f}{\\partial y_i}=\
        \\frac{\\left(\\frac{\\partial f}{\\partial x_i}\\right)}\
        {\\left(\\frac{dy_i}{dx_i}\\right)}.$$
        
        Parameters
        ----------
        untransformed_gradient : numpy.ndarray
            the gradient in the transformed space,
            \\(\\frac{\\partial f}{\\partial y_i}\\)
        untransformed_point : numpy.ndarray
            the input point in the untransformed space, \\(x\\)
        axis : int
            integer index of the axis of the derivative in the gradient array
        
        Returns
        -------
        transformed_gradient : numpy.ndarray
            gradient in transformed space,
            \\(\\frac{\\partial f}{\\partial y_i}\\), as a `numpy.ndarray` of
            same shape as `transformed_gradient`
        """
        gradient = untransformed_gradient.copy()
        axis = (axis % gradient.ndim)
        buffer_slice = ((slice(None),) * axis)
        for (itransform, transform) in enumerate(self.transforms):
            gradient[buffer_slice + (itransform,)] /=\
                transform.derivative(untransformed_point[itransform])
        return gradient
    
    def detransform_hessian(self, transformed_hessian, transformed_gradient,\
        untransformed_point, first_axis=-2):
        """
        Detransforms the Hessian (assuming it was evaluated at the given point)
        from the transformed space to the untransformed space. Assuming
        \\(\\frac{\\partial^2 f}{\\partial y_i\\ \\partial y_j}\\) is the
        Hessian in the transformed space, \\(y\\), and
        \\(\\frac{\\partial f}{\\partial y_i}\\) is the gradient in the
        transformed space, this function encodes the equality:
        $$\\frac{\\partial^2f}{\\partial x_i\\ \\partial x_j}=\
        \\left(\\frac{\\partial^2f}{\\partial y_i\\ \\partial y_j}\\right)\
        \\times\\left(\\frac{dy_i}{dx_i}\\right)\\times\
        \\left(\\frac{dy_j}{dx_j}\\right)+ \\delta_{ij}\\times\
        \\left(\\frac{\\partial f}{\\partial y_i}\\right)\\times\
        \\left(\\frac{d^2y_i}{dx_i^2}\\right),$$ where \\(\\delta_{ij}=\
        \\begin{cases} 1 & i = j \\\\ 0 & i\\ne j \\end{cases}\\).
        
        Parameters
        ----------
        transformed_hessian : numpy.ndarray
            Hessian in the transformed space,
            \\(\\frac{\\partial^2f}{\\partial y_i\\ \\partial y_j}\\)
        transformed_gradient : numpy.ndarray
            gradient in the transformed space,
            \\(\\frac{\\partial f}{\\partial y_i}\\)
        untransformed_point : numpy.ndarray
            the input point in the untransformed space, \\(x\\)
        first_axis : int
            first axis of the derivative in the hessian array (once its mod is
            taken, this should be the same axis of the derivative axis of
            `transformed_gradient`)
        
        Returns
        -------
        untransformed_hessian : numpy.ndarray
            Hessian in the untransformed space,
            \\(\\frac{\\partial^2f}{\\partial x_i\\ \\partial x_j}\\), in a
            `numpy.ndarray` of the same shape as `transformed_hessian`
        """
        hessian = transformed_hessian.copy()
        first_axis = (first_axis % hessian.ndim)
        buffer_slice = ((slice(None),) * first_axis)
        for (itransform, transform) in enumerate(self.transforms):
            location = untransformed_point[itransform]
            derivative = transform.derivative(location)
            second_derivative = transform.second_derivative(location)
            hessian[buffer_slice+(slice(None),itransform)] *= derivative
            hessian[buffer_slice+(itransform,slice(None))] *= derivative
            hessian[buffer_slice+(itransform,itransform)] +=\
                (transformed_gradient[buffer_slice+(itransform,)] *\
                second_derivative)
        return hessian
    
    def transform_hessian(self, untransformed_hessian, transformed_gradient,\
        untransformed_point, first_axis=-2):
        """
        Transforms the hessian (assuming it was evaluated at the given point)
        from the untransformed space to the transformed space. Assuming
        \\(\\frac{\\partial^2f}{\\partial x_i\\ \\partial x_j}\\) is the
        Hessian in the untransformed space, \\(x\\), and
        \\(\\frac{\\partial f}{\\partial y_i}\\) is the gradient in the
        transformed space, this function encodes the equality:
        $$\\frac{\\partial^2f}{\\partial y_i\\ \\partial y_j} =\
        \\frac{\\left(\\frac{\\partial^2f}{\\partial x_i\\ \
        \\partial x_j} \\right)-\\delta_{ij}\\times\
        \\left(\\frac{\\partial f}{\\partial y_i}\\right)\\times\
        \\left(\\frac{d^2y_i}{dx_i^2}\\right)}{\
        \\left(\\frac{dy_i}{dx_i}\\right)\\times\
        \\left(\\frac{dy_j}{dx_j}\\right)},$$ where \\(\\delta_{ij}=\
        \\begin{cases} 1 & i = j \\\\ 0 & i\\ne j \\end{cases}\\).
        
        Parameters
        ----------
        untransformed_hessian : numpy.ndarray
            the hessian in the untransformed space,
            \\(\\frac{\\partial^2f}{\\partial x_i\\ \\partial x_j}\\)
        transformed_gradient : numpy.ndarray
            the gradient in the transformed space,
            \\(\\frac{\\partial f}{\\partial y_i}\\)
        untransformed_point : numpy.ndarray
            the input point in the untransformed space, \\(x\\)
        first_axis : int
            first axis of the derivative in the hessian array (once its mod is
            taken, this should be the same axis of the derivative axis of
            `transformed_gradient`)
        
        Returns
        -------
        transformed_hessian : numpy.ndarray
            Hessian in transformed space,
            \\(\\frac{\\partial^2 f}{\\partial y_i\\ \\partial y_j}\\), as a
            `numpy.ndarray` of same shape as `untransformed_hessian`
        """
        hessian = untransformed_hessian.copy()
        first_axis = (first_axis % hessian.ndim)
        buffer_slice = ((slice(None),) * first_axis)
        for (itransform, transform) in enumerate(self.transforms):
            location = untransformed_point[itransform]
            derivative = transform.derivative(location)
            second_derivative = transform.second_derivative(location)
            hessian[buffer_slice+(itransform,itransform)] -=\
                (transformed_gradient[buffer_slice+(itransform,)] *\
                second_derivative)
            hessian[buffer_slice+(slice(None),itransform)] /= derivative
            hessian[buffer_slice+(itransform,slice(None))] /= derivative
        return hessian
    
    def transform_derivatives(self, untransformed_derivatives,\
        untransformed_point, axis=-2):
        """
        Transforms the given gradient and hessian from the untransformed space
        to the transformed space. For details on how the transformed
        derivatives are computed, see the `TransformList.transform_gradient`
        and `TransformList.transform_hessian` methods.
        
        Parameters
        ----------
        untransformed_derivatives : tuple
            `(untransformed_gradient, untransformed_hessian)` where
            `untransformed_gradient` is the gradient in the untransformed
            space, \\(\\frac{\\partial f}{\\partial x_i}\\), and
            `untransformed_hessian` is the hessian in the untransformed space,
            \\(\\frac{\\partial^2 f}{\\partial x_i\\ \\partial x_j}\\), both
            given as `numpy.ndarray` objects
        untransformed_point : numpy.ndarray
            the point in untransformed space, \\(x\\), at which the gradient
            and hessian were evaluated
        axis : int
            index of axis (which will be modded by the
            `untransformed_hessian.ndim`) corresponding to the derivative axis
            of `untransformed_gradient` and the first derivative dimrension of
            `untransformed_hessian`
        
        Returns
        -------
        transformed_derivatives : tuple
            `(transformed_gradient, transformed_hessian)` where
            `transformed_gradient` and `transformed_hessian` are the gradient,
            \\(\\frac{\\partial f}{\\partial y_i}\\), and the hessian,
            \\(\\frac{\\partial^2f}{\\partial y_i\\ \\partial y_j}\\),
            respectively, in transformed space. They are `numpy.ndarray`
            objects with the same shapes as `untransformed_gradient` and
            `untransformed_hessian`
        """
        (gradient, hessian) = untransformed_derivatives
        axis = (axis % hessian.ndim)
        gradient =\
            self.transform_gradient(gradient, untransformed_point, axis=axis)
        hessian = self.transform_hessian(hessian, gradient,\
            untransformed_point, first_axis=axis)
        return (gradient, hessian)
    
    def detransform_derivatives(self, transformed_derivatives,\
        untransformed_point, axis=-2):
        """
        Detransforms the given gradient and hessian from the transformed space
        to the untransformed space. For details on how the untransformed
        derivatives are computed, see the `TransformList.detransform_gradient`
        and `TransformList.detransform_hessian` methods.
        
        Parameters
        ----------
        transformed_derivatives : tuple
            `(transformed_gradient, transformed_hessian)` where
            `transformed_gradient` is the gradient in the transformed space,
            \\(\\frac{\\partial f}{\\partial y_i}\\), and `transformed_hessian`
            is the hessian in the transformed space,
            \\(\\frac{\\partial^2 f}{\\partial y_i\\ \\partial y_j}\\), both
            given as `numpy.ndarray` objects
        untransformed_point : numpy.ndarray
            the point in untransformed space, \\(x\\), at which the gradient
            and hessian were evaluated
        axis : int
            index of axis (which will be modded by the
            `transformed_hessian.ndim`) corresponding to the derivative axis
            of `transformed_gradient` and the first derivative dimrension of
            `transformed_hessian`
        
        Returns
        -------
        untransformed_derivatives : tuple
            `(untransformed_gradient, untransformed_hessian)` where
            `untransformed_gradient` and `untransformed_hessian` are the
            gradient, \\(\\frac{\\partial f}{\\partial x_i}\\), and the
            hessian, \\(\\frac{\\partial^2f}{\\partial x_i\\ \\partial x_j}\\),
            respectively, in untransformed space. They are `numpy.ndarray`
            objects with the same shapes as `transformed_gradient` and
            `transformed_hessian`
        """
        (gradient, hessian) = transformed_derivatives
        axis = (axis % hessian.ndim)
        hessian = self.detransform_hessian(hessian, gradient,\
            untransformed_point, first_axis=axis)
        gradient =\
            self.detransform_gradient(gradient, untransformed_point, axis=axis)
        return (gradient, hessian)
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with information about this
        `TransformList`.
        
        Parameters
        ----------
        group : h5py.Group
            hdf5 file group to fill with information about this `TransformList`
        """
        for (itransform, transform) in enumerate(self.transforms):
            subgroup = group.create_group('transform_{}'.format(itransform))
            transform.fill_hdf5_group(subgroup)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a `TransformList` object from the given hdf5 file group.
        
        Parameters
        ----------
        group : h5py.Group
            the hdf5 file group from which to load a `TransformList`
        
        Returns
        -------
        loaded_transform_list : `TransformList`
            object loaded from the given hdf5 file group
        """
        transforms = []
        while 'transform_{}'.format(len(transforms)) in group:
            subgroup = group['transform_{}'.format(len(transforms))]
            transforms.append(load_transform_from_hdf5_group(subgroup))
        return TransformList(*transforms)
    
    def __iter__(self):
        """
        Allows for iteration over this `TransformList`.
        
        Returns
        -------
        transform_list : `TransformList`
            `TransformList` objects are their own iterators; so, this method
            returns this object after resetting the internal iteration
            variable.
        """
        self._iteration = 0
        return self
    
    def __next__(self):
        """
        Finds the next element in the iteration over this `TransformList`. This
        method simply calls the `TransformList.next` method.
        
        Returns
        -------
        transform : `distpy.transform.Transform.Transform`
            the next object in this `TransformList`
        """
        return self.next()
    
    def next(self):
        """
        Finds the next element in the iteration over this `TransformList`.
        
        Returns
        -------
        transform : `distpy.transform.Transform.Transform`
            the next object in this `TransformList`
        """
        if self._iteration == self.num_transforms:
            del self._iteration
            raise StopIteration
        to_return = self.transforms[self._iteration]
        self._iteration = self._iteration + 1
        return to_return
    
    def append(self, transform):
        """
        Appends the given object to this `TransformList`.
        
        Parameters
        ----------
        transform : `distpy.transform.Transform.Transform` or str or None
            any object that can be cast to a
            `distpy.transform.Transform.Transform` by the
            `distpy.transform.CastTransform.cast_to_transform` function
        """
        if castable_to_transform(transform):
            self.transforms.append(cast_to_transform(transform))
        else:
            raise TypeError("Given transform was neither a Transform " +\
                "object nor an object which could be cast to a Transform " +\
                "object.")
        if hasattr(self, '_num_transforms'):
            delattr(self, '_num_transforms')
    
    def extend(self, transform_list):
        """
        Extends this `TransformList` by appending the objects stored in
        `transform_list`.
        
        Parameters
        ----------
        transform_list : `TransformList` or sequence
            either a `TransformList` or a sequence of objects that can be cast
            to `distpy.transform.Transform.Transform` objects using the
            `distpy.transform.CastTransform.cast_to_transform` function
        """
        if isinstance(transform_list, TransformList):
            self.transforms.extend(transform_list.transforms)
        elif type(transform_list) in sequence_types:
            transform_list = TransformList(*transform_list)
            self.transforms.extend(transform_list.transforms)
        else:
            raise TypeError("Can only extend TransformList objects with " +\
                "other TransformList objects or by sequences which can be " +\
                "used to initialize a TransformList object.")
        if hasattr(self, '_num_transforms'):
            delattr(self, '_num_transforms')
    
    @property
    def inverse(self):
        """
        A `TransformList` object that stores the inverse of all of the
        `distpy.transform.Transform.Transform` objects in this `TransformList`.
        """
        if not hasattr(self, '_inverse'):
            self._inverse = TransformList(*[invert_transform(transform)\
                for transform in self.transforms])
        return self._inverse
    
    def __add__(self, other):
        """
        "Adds" this `TransformList` to `other` by returning a new
        `TransformList` object with the `distpy.transform.Transform.Transform`
        objects from both lists.
        
        Parameters
        ----------
        other : `TransformList` or `distpy.transform.Transform.Transform` or\
        str or None
            object containing `distpy.transform.Transform.Transform` object(s)
            to add
        
        Returns
        -------
        combined : `TransformList`
            combination of the `distpy.transform.Transform.Transform` objects
            in `self` and `other`
        """
        if isinstance(other, TransformList):
            return TransformList(*(self.transforms + other.transforms))
        elif castable_to_transform(other):
            return\
                TransformList(*(self.transforms + [cast_to_transform(other)]))
        else:
            raise TypeError("The only things which can be added to a " +\
                "TransformList is another TransformList or an object which " +\
                "can be cast to a Transform.")
    
    def __iadd__(self, other):
        """
        "Adds" `other` to this `TransformList` to by appending/extending its
        `distpy.transform.Transform.Transform` objects to this object. Note
        that this does not create a new `TransformList`.
        
        Parameters
        ----------
        other : `TransformList` or `distpy.transform.Transform.Transform` or\
        str or None
            object containing `distpy.transform.Transform.Transform` object(s)
            to add
        
        Returns
        -------
        transform_list : `TransformList`
            this object after `distpy.transform.Transform.Transform` object(s)
            from `other` are added to it
        """
        if isinstance(other, TransformList):
            self.extend(other)
        elif castable_to_transform(other):
            self.append(other)
        else:
            raise TypeError("The only things which can be added to a " +\
                "TransformList is another TransformList or an object which " +\
                "can be cast to a Transform.")
        return self
    
    def __mul__(self, other):
        """
        "Multiplies" this `TransformList` by `other` by forming a new
        `TransformList` of
        `distpy.transform.CompositeTransform.CompositeTransform` objects with
        this object's `distpy.transform.Transform.Transform` objects forming the
        `distpy.transform.CompositeTransform.CompositeTransform.inner_transform`
        objects and `other`'s `distpy.transform.Transform.Transform` objects
        forming the
        `distpy.transform.CompositeTransform.CompositeTransform.outer_transform`
        objects.
        
        Parameters
        ----------
        other : `TransformList
            another `TransformList` object containing the same number of
            `distpy.transform.Transform.Transform` objects
        
        Returns
        -------
        composite_transform_list : `TransformList`
            container filled with combined transformations
        """
        if not isinstance(other, TransformList):
            raise TypeError("TransformList objects can only be multiplied " +\
                "by other TransformList objects.")
        if self.num_transforms != other.num_transforms:
            raise ValueError("TransformList objects can only be multiplied " +\
                "by TransformList objects of the same length.")
        transforms = []
        for (inner, outer) in zip(self.transforms, other.transforms):
            if isinstance(inner, NullTransform):
                transforms.append(outer)
            elif isinstance(outer, NullTransform):
                transforms.append(inner)
            else:
                transforms.append(CompositeTransform(inner, outer))
        return TransformList(*transforms)
    
    def sublist(self, indices):
        """
        Creates a sublist of this `TransformList` corresponding to `indices`.
        
        Parameters
        ----------
        indices : sequence or slice
            object determining which `distpy.transform.Transform.Transform`
            objects to include in the returned value
        
        Returns
        -------
        result : `TransformList`
           new container containing specified elements from this
           `TransformList`
        """
        if isinstance(indices, slice):
            return TransformList(*self.transforms[indices])
        elif type(indices) in sequence_types:
            if all([(type(index) in int_types) for index in indices]):
                return TransformList(*[self.transforms[index]\
                    for index in indices])
            else:
                raise TypeError("Not all indices given were integers.")
        else:
            raise TypeError("indices was set to a non-sequence.")
    
    def __getitem__(self, index):
        """
        Gets a specific element or set of elements from this container.
        
        Parameters
        ----------
        index : int or sequence or slice
            index or indices of the element(s) to retrieve
        
        Returns
        -------
        element : `distpy.transform.Transform.Transform` or `TransformList`
            either single element from this container (if `index` is an `int`)
            or a new `TransformList` containing the specified elements
        """
        if type(index) in int_types:
            return self.transforms[index]
        elif isinstance(index, slice) or (type(index) in sequence_types):
            return self.sublist(index)
        else:
            raise TypeError("index type not recognized.")
    
    @property
    def is_null(self):
        """
        Boolean describing whether this `TransformList` encodes the
        len(self)-length null transformation.
        """
        for transform in self.transforms:
            if not isinstance(transform, NullTransform):
                return False
        return True
    
    def __eq__(self, other):
        """
        Checks if `other` is a `TransformList` with the same elements as this
        one.
        
        Parameters
        ----------
        other : object
            object to check for equality
        
        Returns
        -------
        result : bool
            True if and only if `other` is a `TransformList` with the same
            elements
        """
        if isinstance(other, TransformList):
            if len(self) == len(other):
                for (stransform, otransform) in zip(self, other):
                    if stransform != otransform:
                        return False
                return True
            else:
                return False
        else:
            return False
    
    def __ne__(self, other):
        """
        Checks if `other` is a `TransformList` with the same elements as this
        one.
        
        Parameters
        ----------
        other : object
            object to check for inequality
        
        Returns
        -------
        result : bool
            False if and only if `other` is a `TransformList` with the same
            elements
        """
        return (not self.__eq__(other))

