"""
File: distpy/jumping/JumpingDistributionList.py
Author: Keith Tauscher
Date: 27 May 2019

Description: A container which can hold an arbitrary number of jumping
             distributions, each of which can have any number of parameters
             which it describes (as long as the specific jumping distribution
             supports that number of parameters). JumpingDistribution objects
             can be added through
             JumpingDistributionList.add_distribution(distribution, transforms)
             where distribution is a JumpingDistribution and transforms is a
             TransformList (or something which can be cast into one) describing
             how parameters are transformed. Once all the jumping distributions
             are added, points can be drawn using
             JumpingDistributionList.draw(source) and the log_value of the
             entire list of jumping distributions can be evaluated at a point
             using JumpingDistributionList.log_value(point) just like any other
             JumpingDistribution object. See documentation of individual
             functions for further details.
"""
import numpy as np
import numpy.random as rand
from ..util import int_types, numerical_types, sequence_types
from ..transform import TransformList, NullTransform
from .JumpingDistribution import JumpingDistribution

class JumpingDistributionList(JumpingDistribution):
    """
    An object which keeps track of many jumping distributions which can be
    univariate or multivariate. It provides methods like log_value, which calls
    log_value on all of its constituent distributions, and draw, which draws
    from all of its constituent distributions.
    """
    def __init__(self, jumping_distribution_tuples=[]):
        """
        Creates a new JumpingDistributionList with the given distributions
        inside.
        
        jumping_distribution_tuples: a list of lists/tuples of the form
                                     (jumping_distribution, transforms) where
                                     jumping_distribution is an instance of the
                                     JumpingDistribution class and transforms
                                     is a TransformList (or something which can
                                     be cast into one) which apply to the
                                     parameters of jumping_distribution
        """
        self._data = []
        if type(jumping_distribution_tuples) in sequence_types:
            for idistribution in range(len(jumping_distribution_tuples)):
                this_tup = jumping_distribution_tuples[idistribution]
                if (type(this_tup) in sequence_types):
                    self.add_distribution(*this_tup)
                else:
                    raise ValueError("One of the distribution tuples " +\
                        "provided to the initializer of a " +\
                        "JumpingDistributionList was not a sequence of " +\
                        "length 2 like (jumping_distribution, transforms).")
        else:
            raise ValueError("The jumping_distribution_tuples argument " +\
                "given to the initializer was not list-like. It should be " +\
                "a list of tuples of the form (jumping_distribution, " +\
                "transformations) where jumping_distribution is a " +\
                "JumpingDistribution object and transformations are lists " +\
                "of strings.")
    
    @property
    def empty(self):
        """
        Finds whether this JumpingDistributionList is empty.
        
        returns True if no jumping distributions have been added
                False otherwise
        """
        return (len(self._data) == 0)
    
    @property
    def numparams(self):
        """
        Property storing the number of parameters in this DistributionList.
        """
        return len(self.transform_list)

    def add_distribution(self, jumping_distribution, transforms=None):
        """
        Adds a jumping distribution and any transforms which go with it to the
        JumpingDistributionList.
        
        jumping_distribution: JumpingDistribution object describing the given
                              variates
        transforms: TransformList object (or something castable to one, such as
                    a sequence of strings which can be cast to Transform
                    objects) which apply to the variates (can be a single
                    string if the jumping distribution is univariate)
        """
        if isinstance(jumping_distribution, JumpingDistribution):
            transforms = TransformList.cast(transforms,\
                num_transforms=jumping_distribution.numparams)
            self._data.append((jumping_distribution, transforms))
        else:
            raise ValueError("The jumping_distribution given to a " +\
                "JumpingDistributionList was not recognized as a " +\
                "jumping_distribution.")
    
    @property
    def num_jumping_distributions(self):
        """
        Property storing the number of jumping_distributions in this
        JumpingDistributionList.
        """
        return len(self._data)
    
    def __add__(self, other):
        """
        Adds this JumpingDistributionList to another.
        
        other: a JumpingDistributionList object with parameters distinct from
               the parameters of self
        
        returns: JumpingDistributionList object which is the combination of the
                 given JumpingDistributionList objects
        """
        if isinstance(other, JumpingDistributionList):
            return JumpingDistributionList(\
                jumping_distribution_tuples=self._data+other._data)
        else:
            raise TypeError("Can only add JumpingDistributionList objects " +\
                "to other JumpingDistributionList objects.")
    
    def __iadd__(self, other):
        """
        Adds all jumping_distributions from other to this
        JumpingDistributionList.
        
        other: JumpingDistributionList object
        """
        if isinstance(other, JumpingDistributionList):
            for distribution_tuple in other._data:
                self.add_distribution(*distribution_tuple)
            return self
        else:
            raise TypeError("JumpingDistributionList objects can only have " +\
                "other JumpingDistributionList objects added to them.")
    
    def modify_transforms(self, new_transform_list):
        """
        Finds a JumpingDistributionList with the same jumping distributions but
        different transforms. Draws from this JumpingDistributionList and the
        returned JumpingDistributionList will differ by the given transforms.
        
        new_transform_list: TransformList (or something that can be cast into
                            one) containing the new transforms
        
        returns: new JumpingDistributionList object with the same jumping
                 distribution but different transforms
        """
        new_transform_list =\
            TransformList.cast(transforms, num_transforms=self.numparams)
        (new_data, running_index) = ([], 0)
        for (jumping_distribution, transforms) in self._data:
            new_running_index = running_index + jumping_distribution.numparams
            new_transforms =\
                new_transform_list[running_index:new_running_index]
            running_index = new_running_index
            new_data.append((jumping_distribution, new_transforms))
        return JumpingDistributionList(jumping_distribution_tuples=new_data)
    
    def draw(self, source, shape=None, random=rand):
        """
        Draws a point from this jumping distribution by drawing points from all
        component jumping distributions.
        
        source: the source point in the full space in the form of a 1D
                numpy.ndarray of length numparams.
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
        if self.numparams == 1:
            source = np.array([source])
        for (jumping_distribution, transforms) in self._data:
            numparams = jumping_distribution.numparams
            source_subset = source[params_included:params_included+numparams]
            if numparams == 1:
                transform = transforms[0]
                point[...,params_included] = transforms[0].apply_inverse(\
                    jumping_distribution.draw(transforms[0](source_subset[0]),\
                    shape=shape, random=random))
            else:
                this_draw =\
                    jumping_distribution.draw(np.array([transform(value) for\
                    (value, transform) in zip(source_subset, transforms)]),\
                    shape=shape, random=random)
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
    
    def log_value(self, source, destination):
        """
        Evaluates the log of the product of the values of the jumping
        distributions contained in this JumpingDistributionList.
        
        source: a numparams-length 1D numpy.ndarray representing the source
                point
        destination: a numparams-length 1D numpy.ndarray representing the
                     destination point
        
        returns: the total log_value coming from contributions from all
                 jumping distributions
        """
        if (type(source) in numerical_types) and\
            (type(destination) in numerical_types):
            source = np.array([source])
            destination = np.array([destination])
        if (type(source) in sequence_types) and\
            (type(destination) in sequence_types):
            source = np.array(source)
            destination = np.array(destination)
            if (source.shape == (self.numparams,)) and\
                 (destination.shape == (self.numparams,)):
                result = 0.
                params_included = 0
                for (jumping_distribution, transforms) in self._data:
                    numparams = jumping_distribution.numparams
                    if numparams == 1:
                        transformed_subsource =\
                            transforms[0].apply(source[params_included])
                        transformed_subdestination =\
                            transforms[0].apply(destination[params_included])
                    else:
                        transformed_subsource = [transform.apply(\
                            source[params_included+itransform])\
                            for (itransform, transform) in\
                            enumerate(transforms)]
                        transformed_subdestination = [transform.apply(\
                            destination[params_included+itransform])\
                            for (itransform, transform) in\
                            enumerate(transforms)]
                    result += jumping_distribution.log_value(\
                        transformed_subsource, transformed_subdestination)
                    for (itransform, transform) in enumerate(transforms):
                        result += transform.log_derivative(\
                            destination[params_included+itransform])
                    params_included += numparams
                    if not np.isfinite(result):
                        return -np.inf
                return result
            else:
                raise ValueError("source or destination given to log_value " +\
                    "function of a JumpingDistributionList did not have " +\
                    "the correct length.")
        else:
            raise ValueError("source or destination given to log_value " +\
                "function of a JumpingDistributionList was not an array.")
    
    def log_value_difference(self, source, destination):
        """
        Evaluates the log of the product of the values of the jumping
        distributions contained in this JumpingDistributionList.
        
        source: a numparams-length 1D numpy.ndarray representing the source
                point
        destination: a numparams-length 1D numpy.ndarray representing the
                     destination point
        
        returns: the total log_value coming from contributions from all
                 jumping distributions
        """
        if (type(source) in numerical_types) and\
            (type(destination) in numerical_types):
            source = np.array([source])
            destination = np.array([destination])
        if (type(source) in sequence_types) and\
            (type(destination) in sequence_types):
            source = np.array(source)
            destination = np.array(destination)
            if (source.shape == (self.numparams,)) and\
                 (destination.shape == (self.numparams,)):
                result = 0.
                params_included = 0
                for (jumping_distribution, transforms) in self._data:
                    numparams = jumping_distribution.numparams
                    if numparams == 1:
                        transformed_subsource =\
                            transforms[0].apply(source[params_included])
                        transformed_subdestination =\
                            transforms[0].apply(destination[params_included])
                    else:
                        transformed_subsource = [transform.apply(\
                            source[params_included+itransform])\
                            for (itransform, transform) in\
                            enumerate(transforms)]
                        transformed_subdestination = [transform.apply(\
                            destination[params_included+itransform])\
                            for (itransform, transform) in\
                            enumerate(transforms)]
                    result += jumping_distribution.log_value(\
                        transformed_subsource, transformed_subdestination)
                    for (itransform, transform) in enumerate(transforms):
                        result += transform.log_derivative(\
                            destination[params_included+itransform])
                        result -= transform.log_derivative(\
                            source[params_included+itransform])
                    params_included += numparams
                    if not np.isfinite(result):
                        return -np.inf
                return result
            else:
                raise ValueError("source or destination given to log_value " +\
                    "function of a JumpingDistributionList did not have " +\
                    "the correct length.")
        else:
            raise ValueError("source or destination given to log_value " +\
                "function of a JumpingDistributionList was not an array.")
    
    def __getitem__(self, which):
        """
        Gets a JumpingDistributionList with only the specified jumping
        distributions.
        
        which: either an int, slice, or sequence of ints
        
        returns: a JumpingDistributionList object
        """
        if type(which) in int_types:
            jumping_distribution_list = JumpingDistributionList(\
                jumping_distribution_tuples=[self._data[which]])
        elif isinstance(which, slice):
            jumping_distribution_list = JumpingDistributionList(\
                jumping_distribution_tuples=self._data[which])
        elif type(which) in sequence_types:
            jumping_distribution_list = JumpingDistributionList()
            for which_element in which:
                jumping_distribution_list.add_distribution(\
                    *self._data[which_element])
        else:
            raise ValueError("Only integers, sequences of integers, and " +\
                "slices are allowed as arguments to __getitem__.")
        return jumping_distribution_list
    
    def delete_distribution(self, index):
        """
        Deletes a jumping distribution from this JumpingDistributionList.
        
        index: the integer index of the jumping_distribution to delete
        """
        self._data = self._data[:index] + self._data[index+1:]
    
    def __delitem__(self, which):
        """
        Deletes the specified jumping distributions. For documentation, see
        delete_distribution function.
        
        which: either an integer, a slice, or a sequence of integers
        """
        if type(which) in int_types:
            self.delete_distribution(which)
        elif isinstance(which, slice) or (type(which) in sequence_types):
            if isinstance(which, slice):
                which =\
                    list(range(*which.indices(self.num_jumping_distributions)))
            which = sorted(which)[-1::-1]
            for index in which:
                self.delete_distribution(index)
        else:
            raise ValueError("Only integers, sequences of integers, and " +\
                "slices are allowed as arguments to __delitem__.")
    
    @property
    def is_discrete(self):
        """
        Property storing whether all of the jumping distributions in this list
        are discrete.
        """
        return all([jumping_distribution.is_discrete\
            for (jumping_distribution, transforms) in self._data])
    
    def __eq__(self, other):
        """
        Checks for equality of this JumpingDistributionList with other. Returns
        True if other has the same jumping_distribution_tuples and False
        otherwise.
        """
        def jumping_distribution_tuples_equal(first, second):
            #
            # Checks whether two jumping_distribution_tuple's are equal.
            # Returns True if the distribution and transforms stored in first
            # are the same as those stored in second and False otherwise.
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
        if isinstance(other, JumpingDistributionList):
            if len(self._data) == len(other._data):
                return all([jumping_distribution_tuples_equal(stuple, otuple)\
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
        for (jumping_distribution, transforms) in self._data:
            answer += transforms
        return answer
    
    def discrete_sublist(self):
        """
        Function which compiles a sublist of the JumpingDistribution objects
        in this JumpingDistributionList: those that represent discrete
        variables.
        
        returns: a JumpingDistributionList object containing all
                 JumpingDistribution objects in this JumpingDistributionList
                 which describe discrete variables
        """
        answer = JumpingDistributionList()
        for (jumping_distribution, transforms) in self._data:
            if jumping_distribution.is_discrete:
                answer.add_distribution(jumping_distribution, transforms)
        return answer
    
    def continuous_sublist(self):
        """
        Function which compiles a sublist of the JumpingDistribution objects in
        this JumpingDistributionList: those that represent continuous
        variables.
        
        returns: a JumpingDistributionList object containing all 
                 JumpingDistribution objects in this JumpingDistributionList
                 which describe continuous variables
        """
        answer = JumpingDistributionList()
        for (jumping_distribution, transforms) in self._data:
            if not jumping_distribution.is_discrete:
                answer.add_distribution(jumping_distribution, transforms)
        return answer
    
    def transformed_version(self):
        """
        Function which returns a version of this JumpingDistributionList where
        the parameters exist in transformed space (instead of transforms being
        carried through this object).
        """
        answer = JumpingDistributionList()
        for (jumping_distribution, transforms) in self._data:
            answer.add_distribution(jumping_distribution)
        return answer
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data about this
        JumpingDistributionList. Each jumping distribution tuple is saved as a
        subgroup in the hdf5 file.
        
        group: the hdf5 file group to fill
        """
        group.attrs['class'] = 'JumpingDistributionList'
        for (ituple, jumping_distribution_tuple) in enumerate(self._data):
            (jumping_distribution, transforms) = jumping_distribution_tuple
            subgroup = group.create_group('distribution_{}'.format(ituple))
            jumping_distribution.fill_hdf5_group(subgroup)
            transforms.fill_hdf5_group(subgroup)
    
    @staticmethod
    def load_from_hdf5_group(group, *jumping_distribution_classes):
        """
        Loads a DistributionList object from the given group.
        
        group: the group which was included in self.fill_hdf5_group(group)
        jumping_distribution_classes: JumpingDistribution subclasses with which
                                      to load subdistributions
        
        returns: JumpingDistributionList object
        """
        ituple = 0
        jumping_distribution_tuples = []
        while ('distribution_{}'.format(ituple)) in group:
            subgroup = group['distribution_{}'.format(ituple)]
            jumping_distribution_class_name = subgroup.attrs['class']
            if ituple >= len(jumping_distribution_classes):
                module = __import__('distpy')
                jumping_distribution_class =\
                    getattr(module, jumping_distribution_class_name)
            else:
                jumping_distribution_class =\
                    jumping_distribution_classes[ituple]
            jumping_distribution =\
                jumping_distribution_class.load_from_hdf5_group(subgroup)
            transform_list = TransformList.load_from_hdf5_group(subgroup)
            jumping_distribution_tuples.append(\
                (jumping_distribution, transform_list))
            ituple += 1
        return JumpingDistributionList(\
            jumping_distribution_tuples=jumping_distribution_tuples)

