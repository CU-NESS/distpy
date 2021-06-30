"""
Module containing a container which can hold an arbitrary number of
`distpy.distribution.Distribution.Distribution` objects, each of which can have
any number of parameters which it describes (as long as the specific
`distpy.distribution.Distribution.Distribution` supports that number of
parameters). `distpy.distribution.Distribution.Distribution` objects can be
added through `DistributionList.add_distribution`. Once all the distributions
are added, points can be drawn using the `DistributionList.draw` method and the
log value of the entire set of distributions can be evaluated at a point using
the `DistributionList.log_value` method. See documentation of individual
methods for further details. This class represents a list-like container of
`distpy.distribution.Distribution.Distribution` objects; see
`distpy.distribution.DistributionSet.DistributionSet` for a dictionary- or set-
like container of `distpy.distribution.Distribution.Distribution` objects.
Unlike the `distpy.distribution.DistributionSet.DistributionSet` class,
`DistributionList` is a subclass of
`distpy.distribution.Distribution.Distribution` and implements all of its
methods and properties.

**File**: $DISTPY/distpy/distribution/DistributionList.py  
**Author**: Keith Tauscher  
**Date**: 31 May 2021
"""
import numpy as np
import numpy.random as rand
import scipy.linalg as scila
from ..util import int_types, numerical_types, sequence_types
from ..transform import TransformList, NullTransform
from .Distribution import Distribution

class DistributionList(Distribution):
    """
    An object which keeps track of many distributions which can be univariate
    or multivariate. It provides methods like log_value, which calls log_value
    on all of its constituent distributions, and draw, which draws from all of
    its constituent distributions.
    """
    def __init__(self, distribution_tuples=[]):
        """
        Creates a new `DistributionList` with the given distributions inside.
        
        Parameters
        ----------
        distribution_tuples : sequence
            a list of lists/tuples of the form `(distribution,)` or
            `(distribution, transforms)` where:
            
            - `distribution` is a
            `distpy.distribution.Distribution.Distribution` object
            - `transforms` is either a
            `distpy.transform.TransformList.TransformList` or something that
            can be cast to one (see the
            `distpy.transform.TransformList.TransformList.cast` method). It
            should describe the space in which the distribution applies. For
            example, to draw a variable from a normal distribution in log10
            space, `distribution` should be a
            `distpy.distribution.GaussianDistribution.GaussianDistribution` and
            `transforms` should be a
            `distpy.transform.Log10Transform.Log10Transform`. The result will
            contain only positive numbers.
        """
        self._data = []
        if type(distribution_tuples) in sequence_types:
            for idistribution in range(len(distribution_tuples)):
                this_tup = distribution_tuples[idistribution]
                if (type(this_tup) in sequence_types):
                    self.add_distribution(*this_tup)
                else:
                    raise ValueError("One of the distribution tuples " +\
                        "provided to the initializer of a DistributionList " +\
                        "was not a sequence of length 2 like " +\
                        "(distribution, transforms).")
        else:
            raise ValueError("The distribution_tuples argument given to " +\
                "the initializer was not list-like. It should be a list of " +\
                "tuples of the form (distribution, transformations) " +\
                "where distribution is a Distribution object and " +\
                "transformations are lists of strings.")

    @property
    def empty(self):
        """
        Boolean describing whether this `DistributionList` is empty.
        """
        return (len(self._data) == 0)

    @property
    def numparams(self):
        """
        The total number of parameters described by in this `DistributionList`.
        """
        return len(self.transform_list)
    
    @property
    def mean(self):
        """
        The approximate mean of this distribution. If the transform has a large
        second derivative at the mean, then this approximation is poor.
        """
        if not hasattr(self, '_mean'):
            mean = []
            for (distribution, transforms) in self._data:
                if distribution.numparams == 1:
                    mean.append(transforms[0].apply_inverse(distribution.mean))
                else:
                    this_mean = distribution.mean
                    for (itransform, transform) in enumerate(transforms):
                        mean.append(\
                            transform.apply_inverse(this_mean[itransform]))
            self._mean = np.array(mean)
        return self._mean
    
    @property
    def variance(self):
        """
        The covariance of this distribution.
        """
        if not hasattr(self, '_variance'):
            variances = []
            for (distribution, transforms) in self._data:
                if distribution.numparams == 1:
                    this_mean = np.array([distribution.mean])
                    this_covariance = np.array([[distribution.variance]])
                    variances.append(transforms.inverse.transform_covariance(\
                        this_covariance, this_mean))
                else:
                    this_mean = distribution.mean
                    this_covariance = distribution.variance
                    variances.append(transforms.inverse.transform_covariance(\
                        this_covariance, this_mean))
            self._variance = scila.block_diag(*variances)
        return self._variance

    def add_distribution(self, distribution, transforms=None):
        """
        Adds a `distpy.distribution.Distribution.Distribution` and the
        parameters it describes to the `DistributionList`.
        
        Parameters
        ----------
        distribution : `distpy.distribution.Distribution.Distribution`
            the distribution to add
        transforms : `distpy.transform.TransformList.TransformList` or\
        `distpy.transform.Transform.Transform` or sequence or str or None
            a `distpy.transform.TransformList.TransformList` object (or
            something castable to one, see
            `distpy.transform.TransformList.TransformList.cast`) which apply to
            the parameters (can be a single
            `distpy.transform.Transform.Transform` or something castable to
            one, see `distpy.transform.CastTransform.cast_to_transform`, if the
            distribution is univariate). If `transforms` is None, then the
            transforms are assumed to be
            `distpy.transform.NullTransform.NullTransform`
        """
        if isinstance(distribution, Distribution):
            transforms = TransformList.cast(transforms,\
                num_transforms=distribution.numparams)
            self._data.append((distribution, transforms))
        else:
            raise ValueError("The distribution given to a DistributionList " +\
                "was not recognized as a distribution.")
    
    @property
    def num_distributions(self):
        """
        The number of distributions stored in this `DistributionList` object.
        """
        return len(self._data)
    
    def __add__(self, other):
        """
        Adds this `DistributionList` to another to create a combined set.
        
        Parameters
        ----------
        other : `DistributionList`
            another `DistributionList` with parameters distinct from this one
        
        Returns
        -------
        sum : `DistributionList`
            the combination of the two `DistributionList` objects being added
        """
        if isinstance(other, DistributionList):
            return DistributionList(distribution_tuples=self._data+other._data)
        else:
            raise TypeError("Can only add DistributionList objects to " +\
                "other DistributionList objects.")
    
    def __iadd__(self, other):
        """
        Adds all distributions from `other` to this `DistributionList`.
        
        Parameters
        ----------
        other : `DistributionList`
            set of distributions to add into this one
        
        Returns
        -------
        enlarged : `DistributionList`
            this `DistributionList` after the distributions from `other` have
            been added in
        """
        if isinstance(other, DistributionList):
            for distribution_tuple in other._data:
                self.add_distribution(*distribution_tuple)
            return self
        else:
            raise TypeError("DistributionList objects can only have other " +\
                "DistributionList objects added to them.")
    
    def modify_transforms(self, new_transform_list):
        """
        Creates a `DistributionList` with the same distribution and parameters
        but different transforms. Draws from this `DistributionList` and the
        returned `DistributionList` will differ by the given transforms.
        
        Parameters
        ----------
        new_transform_list : `distpy.transform.TransformList.TransformList` or\
        `distpy.transform.Transform.Transform` or str or None
            a `distpy.transform.TransformList.TransformList` containing the new
            transforms (or something that can be cast to one with a number of
            transforms equal to the `DistributionList.numparams`; see
            `distpy.transform.TransformList.TransformList.cast` for details on
            what can be cast successfully)
        
        Returns
        -------
        modified : `DistributionList`
            new `DistributionList` object with the same distribution and
            parameters but different transforms
        """
        new_transform_list =\
            TransformList.cast(transforms, num_transforms=self.numparams)
        (new_data, running_index) = ([], 0)
        for (distribution, transforms) in self._data:
            new_running_index = running_index + distribution.numparams
            new_transforms =\
                new_transform_list[running_index:new_running_index]
            running_index = new_running_index
            new_data.append((distribution, new_transforms))
        return DistributionList(distribution_tuples=new_data)
    
    def draw(self, shape=None, random=rand):
        """
        Draws a point from all distributions.
        
        Parameters
        ----------
        shape : int or tuple or None
            shape of arrays which are values of return value
        random : `numpy.random.RandomState`
            the random number generator to use (default: numpy.random)
        
        Returns
        -------
        drawn_points : `numpy.ndarray`
            random variates drawn from the distribution (in the following `p`
            is `DistributionList.numparams`):
            - if `shape` is None, then `drawn_points` is a 1D `numpy.ndarray`
            of length `p`
            - if `shape` is an integer, then `drawn_points` is a 2D
            `numpy.ndarray` of shape `(shape,p)`
            - if `shape` is a tuple, then drawn points is a `numpy.ndarray` of
            shape `shape + (p,)`
        """
        none_shape = (type(shape) is type(None))
        if none_shape:
            shape = 1
        if type(shape) in int_types:
            shape = (shape,)
        point = np.ndarray(shape+(self.numparams,))
        params_included = 0
        for (distribution, transforms) in self._data:
            numparams = distribution.numparams
            if numparams == 1:
                transform = transforms[0]
                point[...,params_included] = transforms[0].apply_inverse(\
                    distribution.draw(shape=shape, random=random))
            else:
                this_draw = distribution.draw(shape=shape, random=random)
                for (itransform, transform) in enumerate(transforms):
                    point[...,params_included+itransform] =\
                        transform.apply_inverse(this_draw[...,itransform])
            params_included += numparams
        if self.numparams == 1:
            point = point[...,0]
        if none_shape:
            return point[0]
        else:
            return point
    
    def log_value(self, point):
        """
        Evaluates the log of the product of the values of the
        `distpy.distribution.Distribution.Distribution` objects contained in
        this `DistributionList`, which is the sum of their log values.
        
        Parameters
        ----------
        point : `numpy.ndarray`
            array of parameter values
        
        Returns
        -------
        total_log_value : float
            total log_value coming from contributions from all distributions
        """
        if type(point) in numerical_types:
            point = [point]
        if type(point) in sequence_types:
            point = np.array(point)
            if point.shape == (self.numparams,):
                result = 0.
                params_included = 0
                for (distribution, transforms) in self._data:
                    numparams = distribution.numparams
                    if numparams == 1:
                        subpoint = transforms[0].apply(point[params_included])
                    else:
                        subpoint = [\
                            transform.apply(point[params_included+itransform])\
                            for (itransform, transform) in\
                            enumerate(transforms)]
                    result += distribution.log_value(subpoint)
                    for (itransform, transform) in enumerate(transforms):
                        result += transform.log_derivative(\
                            point[params_included+itransform])
                    params_included += numparams
                    if not np.isfinite(result):
                        return -np.inf
                return result
            else:
                raise ValueError("point given to log_value function of a " +\
                    "DistributionList did not have the correct length.")
        else:
            raise ValueError("point given to log_value function of a " +\
                "DistributionList was not an array.")
    
    def __getitem__(self, which):
        """
        Gets a `DistributionList` with only the specified distributions.
        
        Parameters
        ----------
        which : int or slice or sequence
            the index or indices of the distributions to include in the
            returned value
        
        Returns
        -------
        sublist : `DistributionList`
            a `DistributionList` object with only the specified distribution(s)
        """
        if type(which) in int_types:
            distribution_list =\
                DistributionList(distribution_tuples=[self._data[which]])
        elif isinstance(which, slice):
            distribution_list =\
                DistributionList(distribution_tuples=self._data[which])
        elif type(which) in sequence_types:
            distribution_list = DistributionList()
            for which_element in which:
                distribution_list.add_distribution(*self._data[which_element])
        else:
            raise ValueError("Only integers, sequences of integers, and " +\
                "slices are allowed as arguments to __getitem__.")
        return distribution_list
    
    def delete_distribution(self, index):
        """
        Deletes a distribution from this `DistributionList`.
        
        Parameters
        ----------
        index : int
            the index of the distribution to delete
        """
        self._data = self._data[:index] + self._data[index+1:]
    
    def __delitem__(self, which):
        """
        Deletes a distribution from this `DistributionList`. Alias of
        `DistributionList.delete_distribution` that allows for usage of the
        `del` keyword.
        
        Parameters
        ----------
        index : int
            the index of the distribution to delete
        """
        if type(which) in int_types:
            self.delete_distribution(which)
        elif isinstance(which, slice) or (type(which) in sequence_types):
            if isinstance(which, slice):
                which = list(range(*which.indices(self.num_distributions)))
            which = sorted(which)[-1::-1]
            for index in which:
                self.delete_distribution(index)
        else:
            raise ValueError("Only integers, sequences of integers, and " +\
                "slices are allowed as arguments to __delitem__.")
    
    @property
    def summary_string(self):
        """
        A string with the dimenstionality of the distribution.
        """
        return '{:d}D DistributionList'.format(self.numparams)
    
    @property
    def minimum(self):
        """
        The minimum allowable value(s) in this distribution.
        """
        if not hasattr(self, '_minimum'):
            self._minimum = []
            for (distribution, transforms) in self._data:
                if distribution.numparams == 1:
                    self._minimum.append(transforms[0].untransform_minimum(\
                        distribution.minimum))
                else:
                    self._minimum.extend([transform.untransform_minimum(\
                        minimum) for (minimum, transform) in\
                        zip(distribution.minimum, transforms)])
        return self._minimum
    
    @property
    def maximum(self):
        """
        The maximum allowable value(s) in this distribution.
        """
        if not hasattr(self, '_maximum'):
            self._maximum = []
            for (distribution, transforms) in self._data:
                if distribution.numparams == 1:
                    self._maximum.append(transforms[0].untransform_maximum(\
                        distribution.maximum))
                else:
                    self._maximum.extend([transform.untransform_maximum(\
                        maximum) for (maximum, transform) in\
                        zip(distribution.maximum, transforms)])
        return self._maximum
    
    @property
    def is_discrete(self):
        """
        Boolean describing whether all of the distributions in this list are
        discrete.
        """
        return all([distribution.is_discrete\
            for (distribution, transforms) in self._data])
    
    @staticmethod
    def _distribution_tuples_equal(first, second):
        """
        Checks whether two distribution tuples are equal.
        
        Parameters
        ----------
        first : tuple
            tuple of form `(distribution, parameters)` as internally
            represented in a `DistributionList`
        second : tuple
            tuple of form `(distribution, transforms)` as internally
            represented in a `DistributionList`
        
        Returns
        -------
        result : bool
            True if and only if the distribution and transformations stored in
            `first` are the same as those stored in `second`.
        """
        (first_distribution, first_transforms) = first
        (second_distribution, second_transforms) = second
        numparams = first_distribution.numparams
        if second_distribution.numparams == numparams:
            for (first_transform, second_transform) in\
                zip(first_transforms, second_transforms):
                if first_transform != second_transform:
                    return False
            return (first_distribution == second_distribution)
        else:
            return False
    
    def __eq__(self, other):
        """
        Checks for equality of this `DistributionList` with `other`.
        
        Parameters
        ----------
        other : object
            object to check for equality
        
        Returns
        -------
        result : bool
            True if and only if `other` is a `DistributionList` with the same
            distribution tuples (though they need not be internally stored in
            the same order).
        """
        if isinstance(other, DistributionList):
            if len(self._data) == len(other._data):
                return all([DistributionList._distribution_tuples_equal(\
                    *tuples) for tuples in zip(self._data, other._data)])
            else:
                return False
        else:
            return False
    
    @property
    def transform_list(self):
        """
        The `distpy.transform.TransformList.TransformList` object describing
        the transforms in this `DistributionList`.
        """
        answer = TransformList()
        for (distribution, transforms) in self._data:
            answer += transforms
        return answer
    
    def discrete_sublist(self):
        """
        Compiles the subset of the `Distribution` objects in this
        `DistributionList` that represent discrete variables.
        
        Returns
        -------
        subset : `DistributionList`
            a `DistributionList` object containing all
            `distpy.distribution.Distribution.Distribution` objects in this
            `DistributionList` which describe discrete variables
        """
        answer = DistributionList()
        for (distribution, transforms) in self._data:
            if distribution.is_discrete:
                answer.add_distribution(distribution, transforms)
        return answer
    
    def continuous_sublist(self):
        """
        Compiles the subset of the `Distribution` objects in this
        `DistributionList` that represent continuous variables.
        
        Returns
        -------
        subset : `DistributionList`
            a `DistributionList` object containing all
            `distpy.distribution.Distribution.Distribution` objects in this
            `DistributionList` which describe continuous variables
        """
        answer = DistributionList()
        for (distribution, transforms) in self._data:
            if not distribution.is_discrete:
                answer.add_distribution(distribution, transforms)
        return answer
    
    def transformed_version(self):
        """
        Compiles a version of this `DistributionList` where the parameters
        exist in transformed space (instead of transforms being carried through
        this object).
        
        Returns
        -------
        transformless : `DistributionList`
            a `DistributionList` with the same distributions and parameter
            names but without transforms
        """
        answer = DistributionList()
        for (distribution, transforms) in self._data:
            answer.add_distribution(distribution)
        return answer
    
    def fill_hdf5_group(self, group, save_metadata=True):
        """
        Fills the given hdf5 file group with data about this
        `DistributionList`.
        
        Parameters
        ----------
        group : h5py.Group
            the hdf5 file group to fill
        save_metadata : bool
            - if True, attempts to save metadata alongside distribution list
            and throws error if it fails
            - if False, metadata is ignored in saving process
        """
        group.attrs['class'] = 'DistributionList'
        for (ituple, distribution_tuple) in enumerate(self._data):
            (distribution, transforms) = distribution_tuple
            subgroup = group.create_group('distribution_{}'.format(ituple))
            distribution.fill_hdf5_group(subgroup, save_metadata=save_metadata)
            transforms.fill_hdf5_group(subgroup)
    
    @staticmethod
    def load_from_hdf5_group(group, *distribution_classes):
        """
        Loads a `DistributionList` object from the given group.
        
        Parameters
        ----------
        group : h5py.Group
            the hdf5 file group in which a `DistributionList` was saved
        distribution_classes : sequence
            sequence of Distribution subclasses with which to load
            subdistributions (if the sub-distributions are defined in `distpy`,
            then these are not necessary because all distributions save their
            class as an attribute when they run
            `distpy.distribution.Distribution.Distribution.fill_hdf5_group`)
        
        Returns
        -------
        loaded : `DistributionList`
            a `DistributionList` that was saved in `group`
        """
        ituple = 0
        distribution_tuples = []
        while ('distribution_{}'.format(ituple)) in group:
            subgroup = group['distribution_{}'.format(ituple)]
            distribution_class_name = subgroup.attrs['class']
            if ituple >= len(distribution_classes):
                module = __import__('distpy')
                distribution_class = getattr(module, distribution_class_name)
            else:
                distribution_class =  distribution_classes[ituple]
            distribution = distribution_class.load_from_hdf5_group(subgroup)
            transform_list = TransformList.load_from_hdf5_group(subgroup)
            distribution_tuples.append((distribution, transform_list))
            ituple += 1
        return DistributionList(distribution_tuples=distribution_tuples)
    
    @property
    def gradient_computable(self):
        """
        Boolean describing whether the gradient of the distributions inside
        this `DistributionList` have been implemented.
        """
        answer = True
        for (distribution, transforms) in self._data:
            answer = (answer and distribution.gradient_computable)
        return answer
    
    def gradient_of_log_value(self, point):
        """
        Computes the derivatives of the log value with respect to the
        parameters.
        
        Parameters
        ----------
        point : `numpy.ndarray`
            array of parameter values
        
        Returns
        -------
        gradient : `numpy.ndarray`
            1D array of length `DistributionSet.numparams` of derivative values
            corresponding to the parameters
        """
        if type(point) in sequence_types:
            point = np.array(point)
            result = np.zeros((self.numparams,))
            params_included = 0
            for (idistribution, distribution_tuple) in enumerate(self._data):
                (distribution, transforms) = distribution_tuple
                numparams = distribution.numparams
                if numparams == 1:
                    result[params_included] +=\
                        distribution.gradient_of_log_value(\
                        transforms[0].apply(point[params_included]))
                else:
                    subpoint = np.array([\
                        transform.apply(point[params_included+itransform])\
                        for (itransform, transform) in enumerate(transforms)])
                    result[params_included:params_included+numparams] +=\
                        distribution.gradient_of_log_value(subpoint)
                for index in range(numparams):
                    result[params_included+index] +=\
                        transforms[index].derivative_of_log_derivative(\
                        point[params_included+index])
                params_included += numparams
            return result
        else:
            raise ValueError("point given to gradient_of_log_value " +\
                "function of a DistributionList was not a sequence of values.")
    
    @property
    def hessian_computable(self):
        """
        Boolean describing whether the hessian of the distributions inside this
        `DistributionList` have been implemented.
        """
        answer = True
        for (distribution, transforms) in self._data:
            answer = (answer and distribution.hessian_computable)
        return answer
    
    def hessian_of_log_value(self, point):
        """
        Computes the second derivatives of the log value with respect to the
        parameters.
        
        Parameters
        ----------
        point : `numpy.ndarray`
            array of parameter values
        
        Returns
        -------
        gradient : `numpy.ndarray`
            1D array of length `DistributionSet.numparams` of second derivative
            values corresponding to the parameters
        """
        if type(point) in sequence_types:
            point = np.array(point)
            result = np.zeros((self.numparams,) * 2)
            params_included = 0
            for (idistribution, distribution_tuple) in enumerate(self._data):
                (distribution, transforms) = distribution_tuple
                numparams = distribution.numparams
                if numparams == 1:
                    subpoint = transforms[0].apply(point[params_included])
                    result[params_included,params_included] +=\
                        distribution.hessian_of_log_value(subpoint)
                else:
                    subpoint = np.array([\
                        transform.apply(point[params_included+itransform])\
                        for (itransform, transform) in enumerate(transforms)])
                    result_slice = 2 *\
                        (slice(params_included, params_included + numparams),)
                    result[result_slice] +=\
                        distribution.hessian_of_log_value(subpoint)
                for i in range(numparams):
                    result[params_included+index,params_included+index] +=\
                        transforms[index].second_derivative_of_log_derivative(\
                        point[params_included+index])
                params_included += numparams
            return result
        else:
            raise ValueError("point given to hessian_of_log_value " +\
                "function of a DistributionList was not a sequence of values.")
    
    def copy(self):
        """
        Finds a deep copy of this `DistributionList`.
        
        Returns
        -------
        copied : `DistributionList`
            deep copy of this `DistributionList`
        """
        copied = DistributionList()
        for (distribution, transforms) in self._data:
            copied_distribution = distribution.copy()
            copied_transforms = [transform for transform in transforms]
            copied.add_distribution(copied_distribution, copied_transforms)
        return copied
    
    def reset(self):
        """
        Resets this distribution. This allows ideal distributions to live
        alongside samples as the same kind of object.
        """
        for (distribution, transforms) in self._data:
            distribution.reset()

