"""
File: distpy/transform/TransformList.py
Author: Keith Tauscher
Date: 27 Jan 2018

Description: File containing a class representing a list of Transform objects.
"""
from ..util import sequence_types, Savable, Loadable
from .NullTransform import NullTransform
from .CompositeTransform import CompositeTransform
from .CastTransform import cast_to_transform, castable_to_transform
from .LoadTransform import load_transform_from_hdf5_group

class TransformList(Savable, Loadable):
    """
    Class representing a list of Transform objects.
    """
    def __init__(self, *transforms):
        """
        Initializes a new TransformList.
        
        transforms: Transform objects or objects which can be cast to Transform
                    objects
        """
        self.transforms = transforms
    
    @property
    def transforms(self):
        """
        Property storing list of Transform objects at the heart of this object.
        """
        if not hasattr(self, '_transforms'):
            raise AttributeError("transforms referenced before it was set.")
        return self._transforms
    
    @transforms.setter
    def transforms(self, value):
        """
        Setter for the Transform objects at the heart of this object.
        
        value: sequence of Transform objects or objects which can be cast to
               Transform objects
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
        Property storing the number of transforms in this TransformList.
        """
        if not hasattr(self, '_num_transforms'):
            self._num_transforms = len(self.transforms)
        return self._num_transforms
    
    def __len__(self):
        """
        Finds the length of this TransformList.
        
        returns: number of Transform objects in this TransformList
        """
        return self.num_transforms
    
    def apply(self, untransformed_point, axis=-1):
        """
        Transforms the given point from the untransformed space to the
        transformed space.
        
        untransformed_point: numpy.ndarray of untransformed point values
        axis: axis of the array corresponding to the list of transforms at the
              heart of this TransformList, default -1
        
        returns: the transformed point in a numpy.ndarray of same shape as
                 untransformed_point
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
        Transforms the given point from the untransformed space to the
        transformed space.
        
        untransformed_point: numpy.ndarray of untransformed point values
        axis: axis of the array corresponding to the list of transforms at the
              heart of this TransformList, default -1
        
        returns: the transformed point in a numpy.ndarray of same shape as
                 untransformed_point
        """
        return self.apply(untransformed_point, axis=axis)
    
    def apply_inverse(self, transformed_point, axis=-1):
        """
        Detransforms the given point from the transformed space to the
        untransformed space.
        
        transformed_point: numpy.ndarray of transformed point values
        axis: axis of the array corresponding to the list of transforms at the
              heart of this TransformList, default -1
        
        returns: the untransformed point in a numpy.ndarray of same shape as
                 transformed_point
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
        Detransforms the given point from the transformed space to the
        untransformed space.
        
        transformed_point: numpy.ndarray of transformed point values
        axis: axis of the array corresponding to the list of transforms at the
              heart of this TransformList, default -1
        
        returns: the untransformed point in a numpy.ndarray of same shape as
                 transformed_point
        """
        return self.apply_inverse(transformed_point, axis=axis)
    
    def detransform_gradient(self, transformed_gradient, untransformed_point,\
        axis=-1):
        """
        Detransforms the gradient (assuming it was evaluated at the given
        point) from the transformed space to the untransformed space.
        
        transformed_gradient: the gradient in the transformed space
        untransformed_point: the input point in the untransformed space
        axis: axis of the derivative in the gradient array
        
        returns: numpy.ndarray of same shape as transformed_gradient
        """
        gradient = transformed_gradient.copy()
        axis = (axis % gradient.ndim)
        buffer_slice = ((slice(None),) * axis)
        for (itransform, transform) in enumerate(self.transforms):
            gradient[buffer_slice + (itransform,)] *=\
                transform.derivative(untransformed_point[itransform])
        return gradient
    
    def transform_gradient(self, untransformed_gradient, untransformed_point,\
        axis=-1):
        """
        Transforms the gradient (assuming it was evaluated at the given
        point) from the untransformed space to the transformed space.
        
        transformed_gradient: the gradient in the transformed space
        untransformed_point: the input point in the untransformed space
        axis: axis of the derivative in the gradient array
        
        returns: numpy.ndarray of same shape as transformed_gradient
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
        Detransforms the hessian (assuming it was evaluated at the given point)
        from the transformed space to the untransformed space.
        
        transformed_hessian: the hessian in the transformed space
        transformed_gradient: the gradient in the transformed space
        untransformed_point: the input point in the untransformed space
        first_axis: first axis of the derivative in the hessian array (once its
                    mod is taken, this should be the same axis of the
                    derivative axis of transformed_gradient)
        
        returns: numpy.ndarray of same shape as transformed_hessian
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
        from the untransformed space to the transformed space.
        
        transformed_hessian: the hessian in the transformed space
        transformed_gradient: the gradient in the transformed space
        untransformed_point: the input point in the untransformed space
        first_axis: first axis of the derivative in the hessian array (once its
                    mod is taken, this should be the same axis of the
                    derivative axis of transformed_gradient)
        
        returns: numpy.ndarray of same shape as transformed_hessian
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
        Detransforms the given gradient and hessian from the transformed space
        to the untransformed space.
        
        untransformed_derivatives: tuple of form (untransformed_gradient,
                                   untransformed_hessian)
                                   where untransformed_gradient is the gradient
                                   in the untransformed space and
                                   untransformed_hessian is the hessian in the
                                   untransformed space
        untransformed_point: the point at which the gradient and hessian were
                             evaluated
        axis: axis (which will be modded by the untransformed_hessian ndim)
              corresponding to the untransformed_gradient's derivative
              dimension and the first derivative dimension of the
              untransformed_hessian
        
        returns: (transformed_gradient, transformed_hessian) with same
                 shape as untransformed_gradient and untransformed_hessian
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
        to the untransformed space.
        
        transformed_derivatives: tuple of form
                                 (transformed_gradient, transformed_hessian)
                                 where transformed_gradient is the gradient in
                                 the transformed space and transformed_hessian
                                 is the hessian in the transformed space
        untransformed_point: the point at which the gradient and hessian were
                             evaluated
        axis: axis (which will be modded by the transformed_hessian ndim)
              corresponding to the transformed_gradient's derivative dimension
              and the first derivative dimension of the transformed_hessian
        
        returns: (detransformed_gradient, detransformed_hessian) with same
                 shape as transformed_gradient and transformed_hessian
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
        TransformList.
        
        group: hdf5 file group to fill with information about this
               TransformList
        """
        for (itransform, transform) in enumerate(self.transforms):
            subgroup = group.create_group('transform_{}'.format(itransform))
            transform.fill_hdf5_group(subgroup)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a TransformList object from the given hdf5 file group.
        
        group: the hdf5 file group from which to load a TransformList
        
        returns: a TransformList object derived from the give hdf5 file group
        """
        transforms = []
        while 'transform_{}'.format(len(transforms)) in group:
            subgroup = group['transform_{}'.format(len(transforms))]
            transforms.append(load_transform_from_hdf5_group(subgroup))
        return TransformList(*transforms)
    
    def __iter__(self):
        """
        Returns an iterator over this TransformList. Since TransformList
        objects are their own iterators, this method returns self after
        resetting the internal iteration variable.
        """
        self._iteration = 0
        return self
    
    def next(self):
        """
        Finds the next element in the iteration over this TransformList.
        
        returns: next Transform object in this TransformList
        """
        if self._iteration == self.num_transforms:
            del self._iteration
            raise StopIteration
        to_return = self.transforms[self._iteration]
        self._iteration = self._iteration + 1
        return to_return
    
    def append(self, transform):
        """
        Appends the given Transform object (or object castable as a transform)
        to this TransformList.
        
        transform: must be a Transform or castable as a transform
        """
        if castable_to_transform(transform):
            self.transforms.append(cast_to_transform(transform))
        else:
            raise TypeError("Given transform was neither a Transform " +\
                "object nor an object which could be cast to a Transform " +\
                "object.")
        del self._num_transforms
    
    def extend(self, transform_list):
        """
        Extends this TransformList by concatenating the Transform objects
        stored in this TransformList with the ones stored in transform_list.
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
        del self._num_transforms
    
    def __add__(self, other):
        """
        "Adds" this TransformList to other by returning a new TransformList
        object with the Transforms in both objects.
        
        other: either a Transform (or something castable as a Transform) or a
               TransformList
        
        returns: a TransformList composed of all of the Transforms in this
                 TransformList as well as the Transform(s) in other
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
        "Adds" other to this TransformList to other by appending/extending its
        transforms to this object. Note that this does not create a new
        TransformList.
        
        other: either a Transform (or something castable as a Transform) or a
               TransformList
        """
        if isinstance(other, TransformList):
            self.extend(other)
        elif castable_to_transform(other):
            self.append(other)
        else:
            raise TypeError("The only things which can be added to a " +\
                "TransformList is another TransformList or an object which " +\
                "can be cast to a Transform.")
    
    def __mul__(self, other):
        """
        "Multiplies" other by this TransformList by forming a new TransformList
        of composite transforms with this TransformList's Transforms forming
        the inner transforms and other's Transforms forming the outer
        transforms.
        
        other: must be a TransformList object with the same number of Transform
               objects
        
        returns: TransformList of combined transforms with this TransformList
                 holding the inner transform and other holding outer transforms
        """
        if not isinstance(other, TransformList):
            raise TypeError("TransformList objects can only be added to " +\
                "other TransformList objects.")
        if self.num_transforms != other.num_transforms:
            raise ValueError("TransformList objects can only be added to " +\
                "TransformList objects of the same length.")
        transforms = []
        for (inner, outer) in zip(self.transforms, other.transforms):
            if isinstance(inner, NullTransform):
                transforms.append(outer)
            elif isinstance(outer, NullTransform):
                transforms.append(inner)
            else:
                transforms.append(CompositeTransform(inner, outer))
        return TransformList(*transforms)
    
    def __getitem__(self, index):
        """
        Gets a specific element of the Transforms sequence.
        
        index: the index of the element to retrieve
        
        returns: a Transform object
        """
        return self.transforms[index]
    
    def __eq__(self, other):
        """
        Checks if other is a TransformList with the same Transforms as this
        one.
        
        other: object to check for equality
        
        returns: True iff other is TransformList with same Transforms
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
        Ensures that (a!=b) == (not (a==b)).
        
        other: object to check for inequality
        
        returns: False iff other is TransformList with same Transforms
        """
        return (not self.__eq__(other))

