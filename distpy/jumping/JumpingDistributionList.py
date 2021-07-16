"""
Module containing a container which can hold an arbitrary number of
`distpy.jumping.JumpingDistribution.JumpingDistribution` objects, each of which
can have any number of parameters which it describes (as long as the specific
`distpy.jumping.JumpingDistribution.JumpingDistribution` supports that number
of parameters). `distpy.jumping.JumpingDistribution.JumpingDistribution`
objects can be added through `JumpingDistributionList.add_distribution`. Once
all the distributions are added, points can be drawn using the
`JumpingDistributionList.draw` method and the log value of the entire set of
distributions can be evaluated at a given source and destination using the
`JumpingDistributionList.log_value` and
`JumpingDistributionList.log_value_difference` methods. See documentation of
individual methods for further details. This class represents a list-like
container of `distpy.jumping.JumpingDistribution.JumpingDistribution` objects;
see `distpy.jumping.JumpingDistributionSet.JumpingDistributionSet` for a
dictionary- or set-like container of
`distpy.jumping.JumpingDistribution.JumpingDistribution` objects. Unlike the
`distpy.jumping.JumpingDistributionSet.JumpingDistributionSet` class,
`JumpingDistributionList` is a subclass of
`distpy.jumping.JumpingDistribution.JumpingDistribution` and implements all of
its methods and properties.

**File**: $DISTPY/distpy/jumping/JumpingDistributionList.py  
**Author**: Keith Tauscher  
**Date**: 3 Jul 2021
"""
import numpy as np
import numpy.random as rand
from ..util import int_types, numerical_types, sequence_types
from ..transform import TransformList, NullTransform
from .JumpingDistribution import JumpingDistribution

class JumpingDistributionList(JumpingDistribution):
    """
    A container which can hold an arbitrary number of
    `distpy.jumping.JumpingDistribution.JumpingDistribution` objects, each of
    which can have any number of parameters which it describes (as long as the
    specific `distpy.jumping.JumpingDistribution.JumpingDistribution` supports
    that number of parameters).
    `distpy.jumping.JumpingDistribution.JumpingDistribution` objects can be
    added through `JumpingDistributionList.add_distribution`. Once all the
    distributions are added, points can be drawn using the
    `JumpingDistributionList.draw` method and the log value of the entire set
    of distributions can be evaluated at a given source and destination using
    the `JumpingDistributionList.log_value` and
    `JumpingDistributionList.log_value_difference` methods. See documentation
    of individual methods for further details. This class represents a
    list-like container of
    `distpy.jumping.JumpingDistribution.JumpingDistribution` objects; see
    `distpy.jumping.JumpingDistributionSet.JumpingDistributionSet` for a
    dictionary- or set-like container of
    `distpy.jumping.JumpingDistribution.JumpingDistribution` objects. Unlike
    the `distpy.jumping.JumpingDistributionSet.JumpingDistributionSet` class,
    `JumpingDistributionList` is a subclass of
    `distpy.jumping.JumpingDistribution.JumpingDistribution` and implements all
    of its methods and properties.
    """
    def __init__(self, jumping_distribution_tuples=[]):
        """
        Creates a new `JumpingDistributionList` with the given distributions
        inside.
        
        Parameters
        ----------
        jumping_distribution_tuples : sequence
            a list of sequences of the form `(distribution, transforms)` or
            `(distribution,)` where:
            
            - `distribution` is a
            `distpy.jumping.JumpingDistribution.JumpingDistribution`object
            - `transforms` (if given) is a
            `distpy.transform.TransformList.TransformList` or something that
            can be cast to one (can be a single
            `distpy.transform.Transform.Transform` if `distribution` is
            univariate)
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
        Boolean describing whether this `JumpingDistributionList` is empty.
        """
        return (len(self._data) == 0)
    
    @property
    def numparams(self):
        """
        The integer number of parameters described by this
        `JumpingDistributionList`.
        """
        return len(self.transform_list)
    
    def __len__(self):
        """
        Function allowing users to access the number of parameters described by
        this `JumpingDistributionList` by using the built-in `len` function and
        not explicitly referencing `JumpingDistributionList.numparams`.
        """
        return self.numparams

    def add_distribution(self, jumping_distribution, transforms=None):
        """
        Adds a `distpy.jumping.JumpingDistribution.JumpingDistribution` and the
        parameters it describes to the `JumpingDistributionList`.
        
        Parameters
        ----------
        jumping_distribution : `distpy.jumping.JumpingDistribution.JumpingDistribution`
            distribution describing how the given parameters jump
        transforms : `distpy.transform.TransformList.TransformList` or\
        `distpy.transform.Transform.Transform` or sequence or None
            list of transformations to apply to the parameters (can be a single
            string if `distribution` is univariate)
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
        The integer number of
        `distpy.jumping.JumpingDistribution.JumpingDistribution` objects in
        this `JumpingDistributionList`.
        """
        return len(self._data)
    
    def __add__(self, other):
        """
        Adds this `JumpingDistributionList` to another by multiplying their
        PDFs and assuming the parameters of `self` and `other` are independent.
        Allows for use of the `+` operator on `JumpingDistributionList`
        objects.
        
        Parameters
        ----------
        other : `JumpingDistributionList`
            a `JumpingDistributionList` to combine with this one
        
        Returns
        -------
        combined : `JumpingDistributionList`
            combination of the given `JumpingDistributionList` objects
        """
        if isinstance(other, JumpingDistributionList):
            return JumpingDistributionList(\
                jumping_distribution_tuples=self._data+other._data)
        else:
            raise TypeError("Can only add JumpingDistributionList objects " +\
                "to other JumpingDistributionList objects.")
    
    def __iadd__(self, other):
        """
        Adds all distributions from `other` to this `JumpingDistributionList`.
        Allows for use of the `+=` operator with `JumpingDistributionList`
        objects.
        
        Parameters
        ----------
        other : `JumpingDistributionList`
            `JumpingDistributionList` with parameters distinct from the
            parameters of `self`
        
        Returns
        -------
        self : `JumpingDistributionList`
            this object with its state changed to include distributions from
            `other`
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
        Finds a `JumpingDistributionList` with the same jumping distributions
        but different transforms. Draws from this `JumpingDistributionList` and
        the returned `JumpingDistributionList` will differ by the given
        transformations.
        
        Parameters
        ----------
        new_transform_list : `distpy.transform.TransformList.TransformList` or\
        `distpy.transform.Transform.Transform` or sequence or None
            list of transformations to apply to the parameters (can be a single
            string if `distribution` is univariate)
        
        Returns
        -------
        modified : `JumpingDistributionList`
            new `JumpingDistributionList` object with the same jumping
            distribution but different transforms
        """
        new_transform_list =\
            TransformList.cast(self.transforms, num_transforms=self.numparams)
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
        Draws a destination from all distributions given the source.
        
        Parameters
        ----------
        source : numpy.ndarray
            1D array of source values
        shape : tuple or int or None
            - if `shape` is None, the values of `destination[...,index]` are
            numbers
            - if `shape` is an int, the values of `destination[...,index]` are
            1D arrays of length `shape`
            - if `shape` is a tuple, the values of `destination[...,index]` are
            arrays of shape `shape`
        
        Returns
        -------
        destination : dict
            array of destination values (shape determined by `shape` parameter)
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
        Evaluates the log of the product of the values of the distributions
        contained in this `JumpingDistributionList` from `source` to
        `destination`.
        
        Parameters
        ----------
        source : numpy.ndarray
            1D array containing source values
        destination : numpy.ndarray
            1D array containing destination values
        
        Returns
        -------
        total_log_value : float
            the total `log_value` coming from contributions from all
            distributions. If `source` is
            \\(\\begin{bmatrix} \\boldsymbol{x}_1 \\\\ \\boldsymbol{x}_2 \\\\\
            \\vdots \\\\ \\boldsymbol{x}_N \\end{bmatrix}\\) and `destination`
            is \\(\\begin{bmatrix} \\boldsymbol{y}_1 \\\\\
            \\boldsymbol{y}_2 \\\\ \\vdots \\\\ \\boldsymbol{y}_N\
            \\end{bmatrix}\\), then `total_log_value` is
            \\(\\sum_{k=1}^N\\ln{g_k(\\boldsymbol{x}_k,\\boldsymbol{y}_k)}\\),
            where \\(g_k\\) is the PDF of the \\(k^{\\text{th}}\\) parameter
            chunk.
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
        Evaluates the log of the product of the ratios of the distributions
        contained in this `JumpingDistributionList` from source to destination.
        
        Parameters
        ----------
        source : numpy.ndarray
            1D array containing source values
        destination : numpy.ndarray
            1D array containing destination values
        
        Returns
        -------
        total_log_value_difference : float
            the total `log_value` coming from contributions from all
            distributions. If `source` is
            \\(\\begin{bmatrix} \\boldsymbol{x}_1 \\\\ \\boldsymbol{x}_2 \\\\\
            \\vdots \\\\ \\boldsymbol{x}_N \\end{bmatrix}\\) and `destination`
            is \\(\\begin{bmatrix} \\boldsymbol{y}_1 \\\\\
            \\boldsymbol{y}_2 \\\\ \\vdots \\\\ \\boldsymbol{y}_N\
            \\end{bmatrix}\\), then `total_log_value_difference` is
            \\(\\sum_{k=1}^N\\left[\\ln{g_k(\\boldsymbol{x}_k,\
            \\boldsymbol{y}_k)}-\\ln{g_k(\\boldsymbol{y}_k,\
            \\boldsymbol{x}_k)}\\right]\\), where \\(g_k\\) is the PDF of the
            \\(k^{\\text{th}}\\) parameter chunk.
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
        Gets a `JumpingDistributionList` with only the specified jumping
        distributions.
        
        Parameters
        ----------
        which : int or sequence or slice
            object determining which distributions to keep
        
        Returns
        -------
        subset : `JumpingDistributionList`
            a JumpingDistributionList object with only the given distributions
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
        Deletes a distribution from this `JumpingDistributionList`.
        
        Parameters
        ----------
        index : int
            the index of the distribution to delete
        """
        self._data = self._data[:index] + self._data[index+1:]
    
    def __delitem__(self, which):
        """
        Deletes a distribution from this `JumpingDistributionList`. Alias for
        `JumpingDistributionList.delete_distribution` that allows for the use
        of the `del` keyword with `JumpingDistributionList` objects
        
        Parameters
        ----------
        which : int or sequence or slice
            object determining which distributions to delete
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
        Boolean describing whether all of the jumping distributions in this
        `JumpingDistributionList` are discrete.
        """
        return all([jumping_distribution.is_discrete\
            for (jumping_distribution, transforms) in self._data])
    
    def __eq__(self, other):
        """
        Checks for equality of this `JumpingDistributionList` with `other`.
        
        Parameters
        ----------
        other : object
            object with which to check for equality
        
        Returns
        -------
        result : bool
            True if and only if other is another `JumpingDistributionList` that
            has the same distribution tuples
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
        Checks for inequality of this `JumpingDistributionList` with `other`.
        
        Parameters
        ----------
        other : object
            object with which to check for inequality
        
        Returns
        -------
        result : bool
            False if and only if other is another `JumpingDistributionList`
            that has the same distribution tuples
        """
        return (not self.__eq__(other))
    
    @property
    def transform_list(self):
        """
        The `distpy.transform.TransformList.TransformList` object describing
        the transformations of the parameters in this
        `JumpingDistributionList`.
        """
        answer = TransformList()
        for (jumping_distribution, transforms) in self._data:
            answer += transforms
        return answer
    
    def discrete_sublist(self):
        """
        Function which compiles a sublist of the
        `distpy.jumping.JumpingDistribution.JumpingDistribution` objects in
        this `JumpingDistributionList`: those that represent discrete
        variables.
        
        Returns
        -------
        sublist : `JumpingDistributionList`
            a `JumpingDistributionList` object containing all
            `distpy.jumping.JumpingDistribution.JumpingDistribution` objects in
            this `JumpingDistributionList` which describe discrete variables
        """
        answer = JumpingDistributionList()
        for (jumping_distribution, transforms) in self._data:
            if jumping_distribution.is_discrete:
                answer.add_distribution(jumping_distribution, transforms)
        return answer
    
    def continuous_sublist(self):
        """
        Function which compiles a sublist of the
        `distpy.jumping.JumpingDistribution.JumpingDistribution` objects in
        this `JumpingDistributionList`: those that represent continuous
        variables.
        
        Returns
        -------
        sublist : `JumpingDistributionList`
            a `JumpingDistributionList` object containing all
            `distpy.jumping.JumpingDistribution.JumpingDistribution` objects in
            this `JumpingDistributionList` which describe continuous variables
        """
        answer = JumpingDistributionList()
        for (jumping_distribution, transforms) in self._data:
            if not jumping_distribution.is_discrete:
                answer.add_distribution(jumping_distribution, transforms)
        return answer
    
    def transformed_version(self):
        """
        Finds a version of this `JumpingDistributionList` that exists in
        transformed space.
        
        Returns
        -------
        transformed : `JumpingDistributionList`
            a version of this `JumpingDistributionList` where the parameters
            exist in transformed space (instead of transforms being carried
            through this object).
        """
        answer = JumpingDistributionList()
        for (jumping_distribution, transforms) in self._data:
            answer.add_distribution(jumping_distribution)
        return answer
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data about this
        `JumpingDistributionList`. Each jumping distribution tuple is saved as
        a subgroup in the hdf5 file.
        
        Parameters
        ----------
        group : h5py.Group
            the hdf5 file group to fill
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
        Loads a `DistributionList` object from the given group.
        
        Parameters
        ----------
        group : h5py.Group
            the group which was included in self.fill_hdf5_group(group)
        jumping_distribution_classes : sequence
            sequence of `distpy.jumping.JumpingDistribution.JumpingDistribution`
            subclasses with which to load subdistributions
        
        Returns
        -------
        loaded: `JumpingDistributionList`
            `JumpingDistributionList` loaded from the given group
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

