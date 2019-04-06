"""
File: distpy/distribution/DistributionList.py
Author: Keith Tauscher
Date: 23 Sep 2018

Description: A container which can hold an arbitrary number of distributions,
             each of which can have any number of parameters which it describes
             (as long as the specific distribution supports that number of
             parameters). Distribution objects can be added through
             DistributionList.add_distribution(distribution, transforms) where
             distribution is a Distribution and transforms is a TransformList
             (or something which can be cast into one) describing how
             parameters are transformed. Once all the distributions are added,
             points can be drawn using DistributionList.draw() and the
             log_value of the entire list of distributions can be evaluated at
             a point using DistributionList.log_value(point) just like any
             other Distribution object. See documentation of individual
             functions for further details.
"""
import numpy as np
import numpy.random as rand
from ..util import int_types, sequence_types
from ..transform import cast_to_transform_list, TransformList, NullTransform
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
        Creates a new DistributionList with the given distributions inside.
        
        distribution_tuples: a list of lists/tuples of the form
                             (distribution, transforms) where distribution is
                             an instance of the Distribution class and
                             transforms is a TransformList (or something which
                             can be cast into one) which apply to the
                             parameters of distribution
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
        Finds whether this DistributionList is empty.
        
        returns True if no distributions have been added, False otherwise
        """
        return (len(self._data) == 0)

    @property
    def numparams(self):
        """
        Property storing the number of parameters in this DistributionList.
        """
        return len(self.transform_list)

    def add_distribution(self, distribution, transforms=None):
        """
        Adds a distribution and any transforms which go with it to the
        DistributionList.
        
        distribution: Distribution object describing the given variates
        transforms: TransformList object (or something castable to one, such as
                    a sequence of strings which can be cast to Transform
                    objects) which apply to the variates (can be a single
                    string if the distribution is univariate)
        """
        if isinstance(distribution, Distribution):
            transforms = cast_to_transform_list(transforms,\
                num_transforms=distribution.numparams)
            self._data.append((distribution, transforms))
        else:
            raise ValueError("The distribution given to a DistributionList " +\
                "was not recognized as a distribution.")
    
    @property
    def num_distributions(self):
        """
        Property storing the number of distributions in this 
        """
        return len(self._data)
    
    def __add__(self, other):
        """
        Adds this DistributionList to another.
        
        other: a DistributionList object with parameters distinct from the
               parameters of self
        
        returns: DistributionList object which is the combination of the given
                 DistributionList objects
        """
        if isinstance(other, DistributionList):
            return DistributionList(distribution_tuples=self._data+other._data)
        else:
            raise TypeError("Can only add DistributionList objects to " +\
                "other DistributionList objects.")
    
    def __iadd__(self, other):
        """
        Adds all distributions from other to this DistributionList.
        
        other: DistributionList object
        """
        if isinstance(other, DistributionList):
            for distribution_tuple in other._data:
                self.add_distribution(*distribution_tuple)
            return self
        else:
            raise TypeError("DistributionList objects can only have other " +\
                "DistributionList objects added to them.")
    
    def with_different_transforms(self, new_transform_list):
        """
        Finds a DistributionList with the same distributions but different
        transforms. Draws from this DistributionList and the returned
        DistributionList will differ by the given transforms.
        
        new_transform_list: TransformList (or something that can be cast into
                            one) containing the new transforms
        
        returns: new DistributionList object with the same distribution but
                 different transforms
        """
        new_transform_list =\
            cast_to_transform_list(transforms, num_transforms=self.numparams)
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
        Draws a point from this distribution by drawing points from all
        component distributions.
        
        shape: shape of arrays which are values of return value
        random: the random number generator to use (default: numpy.random)
        
        returns a numpy.ndarray with shape given by shape+(numparams,)
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
        if none_shape:
            return point[0]
        else:
            return point
    
    def log_value(self, point):
        """
        Evaluates the log of the product of the values of the distributions
        contained in this DistributionList.
        
        point: an numparams-length 1D numpy.ndarray
        
        returns: the total log_value coming from contributions from all
                 distributions
        """
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
        Gets a DistributionList with only the specified distributions.
        
        which: either an int, slice, or sequence of ints
        
        returns: a DistributionList object
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
        Deletes a distribution from this DistributionList.
        
        index: the integer index of the distribution to delete
        """
        self._data = self._data[:index] + self._data[index+1:]
    
    def __delitem__(self, which):
        """
        Deletes the specified distributions. For documentation, see
        delete_distribution function.
        
        which: either an integer, a slice, or a sequence of integers
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
        Property which yields a string with the dimenstionality of the
        distribution.
        """
        return '{:d}D DistributionList'.format(self.numparams)
    
    @property
    def minimum(self):
        """
        Property storing the minimum allowable value(s) in this distribution.
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
        Property storing the maximum allowable value(s) in this distribution.
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
        Property storing whether all of the distributions in this list are
        discrete.
        """
        return all([distribution.is_discrete\
            for (distribution, transforms) in self._data])
    
    def __eq__(self, other):
        """
        Checks for equality of this DistributionList with other. Returns True
        if other has the same distribution_tuples and False otherwise.
        """
        def distribution_tuples_equal(first, second):
            #
            # Checks whether two distribution_tuple's are equal. Returns True
            # if the distribution and transforms stored in first are the same
            # as those stored in second and False otherwise.
            #
            (fdistribution, ftransforms) = first
            (sdistribution, stransforms) = second
            numparams = fdistribution.numparams
            if sdistribution.numparams == numparams:
                for (ftransform, stransform) in zip(ftransforms, stransforms):
                    if ftransform != stransform:
                        return False
                return (fdistribution == sdistribution)
            else:
                return False
        if isinstance(other, DistributionList):
            if len(self._data) == len(other._data):
                return all([distribution_tuples_equal(stuple, otuple)\
                    for (stuple, otuple) in zip(self._data, other._data)])
            else:
                return False
        else:
            return False
    
    def __ne__(self, other):
        """
        This function simply asserts that (a != b) == (not (a == b))
        """
        return (not self.__eq__(other))
    
    @property
    def transform_list(self):
        """
        Property storing the TransformList object describing the transforms in
        this DistributionList.
        """
        answer = TransformList()
        for (distribution, transforms) in self._data:
            answer += transforms
        return answer
    
    def discrete_sublist(self):
        """
        Function which compiles a sublist of the Distribution objects in this
        DistributionList: those that represent discrete variables.
        
        returns: a DistributionList object containing all Distribution objects
                 in this DistributionList which describe discrete variables
        """
        answer = DistributionList()
        for (distribution, transforms) in self._data:
            if distribution.is_discrete:
                answer.add_distribution(distribution, transforms)
        return answer
    
    def continuous_sublist(self):
        """
        Function which compiles a sublist of the Distribution objects in this
        DistributionList: those that represent continuous variables.
        
        returns: a DistributionList object containing all Distribution objects
                 in this DistributionList which describe continuous variables
        """
        answer = DistributionList()
        for (distribution, transforms) in self._data:
            if not distribution.is_discrete:
                answer.add_distribution(distribution, transforms)
        return answer
    
    def transformed_version(self):
        """
        Function which returns a version of this DistributionList where the
        parameters exist in transformed space (instead of transforms being
        carried through this object).
        """
        answer = DistributionList()
        for (distribution, transforms) in self._data:
            answer.add_distribution(distribution)
        return answer
    
    def fill_hdf5_group(self, group, save_metadata=True):
        """
        Fills the given hdf5 file group with data about this DistributionList.
        Each distribution tuple is saved as a subgroup in the hdf5 file.
        
        group: the hdf5 file group to fill
        save_metadata: if True, attempts to save metadata alongside
                                distribution list and throws error if it fails
                       if False, metadata is ignored in saving process
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
        Loads a DistributionList object from the given group.
        
        group: the group which was included in self.fill_hdf5_group(group)
        distribution_classes: Distribution subclasses with which to load
                              subdistributions
        
        returns: DistributionList object
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
        Property which stores whether the gradient of the given distribution
        has been implemented.
        """
        answer = True
        for (distribution, transforms) in self._data:
            answer = (answer and distribution.gradient_computable)
        return answer
    
    def gradient_of_log_value(self, point):
        """
        Computes the derivative(s) of log_value(point) with respect to the
        parameter(s).
        
        point: array of values
        
        returns: 1D numpy.ndarray containing the N derivatives of the log value
                 with respect to each individual parameter
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
        Property which stores whether the hessian of the given distribution
        has been implemented.
        """
        answer = True
        for (distribution, transforms) in self._data:
            answer = (answer and distribution.hessian_computable)
        return answer
    
    def hessian_of_log_value(self, point):
        """
        Computes the second derivative(s) of log_value(point) with respect to
        the parameter(s).
        
        point: array of values
        
        returns: 2D square numpy.ndarray with dimension length equal to the
                 number of parameters representing the N^2 different second
                 derivatives of the log value
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
        Returns a deep copy of this DistributionList.
        """
        copied = DistributionList()
        for (distribution, transforms) in self._data:
            copied_distribution = distribution.copy()
            copied_transforms =\
                [transform.to_string() for transform in transforms]
            copied.add_distribution(copied_distribution, copied_transforms)
        return copied
    
    def reset(self):
        """
        Resets this distribution. This allows ideal distributions to live
        alongside samples as the same kind of object.
        """
        for (distribution, transforms) in self._data:
            distribution.reset()

