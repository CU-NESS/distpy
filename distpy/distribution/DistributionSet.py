"""
Module containing a container which can hold an arbitrary number of
`distpy.distribution.Distribution.Distribution` objects, each of which can have
any number of parameters which it describes (as long as the specific
`distpy.distribution.Distribution.Distribution` supports that number of
parameters). `distpy.distribution.Distribution.Distribution` objects can be
added through `DistributionSet.add_distribution`. Once all the distributions
are added, points can be drawn using the `DistributionSet.draw` method and the
log value of the entire set of distributions can be evaluated at a point using
the `DistributionSet.log_value` method. See documentation of individual methods
for further details. This class represents a dictionary- or set-like container
of `distpy.distribution.Distribution.Distribution` objects; see
`distpy.distribution.DistributionList.DistributionList` for a list-like
container of `distpy.distribution.Distribution.Distribution` objects.

**File**: $DISTPY/distpy/distribution/DistributionSet.py  
**Author**: Keith Tauscher  
**Date**: 30 May 2021
"""
import numpy as np
import numpy.random as rand
from ..util import Savable, Loadable, int_types, sequence_types,\
    univariate_histogram, bivariate_histogram, triangle_plot
from ..transform import NullTransform, TransformList, TransformSet
from .Distribution import Distribution
from .DistributionList import DistributionList
from .LoadDistribution import load_distribution_from_hdf5_group
try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str

class DistributionSet(Savable, Loadable):
    """
    A container which can hold an arbitrary number of
    `distpy.distribution.Distribution.Distribution` objects, each of which can
    have any number of parameters which it describes (as long as the specific
    `distpy.distribution.Distribution.Distribution` supports that number of
    parameters). `distpy.distribution.Distribution.Distribution` objects can be
    added through `DistributionSet.add_distribution`. Once all the
    distributions are added, points can be drawn using the
    `DistributionSet.draw` method and the log value of the entire set of
    distributions can be evaluated at a point using the
    `DistributionSet.log_value` method. See documentation of individual methods
    for further details. This class represents a dictionary- or set-like
    container of `distpy.distribution.Distribution.Distribution` objects; see
    `distpy.distribution.DistributionList.DistributionList` for a list-like
    container of `distpy.distribution.Distribution.Distribution` objects.
    """
    def __init__(self, distribution_tuples=[]):
        """
        Creates a new `DistributionSet` with the given distributions inside.
        
        Parameters
        ----------
        distribution_tuples : sequence
            a list of lists/tuples of the form `(distribution, params)` or
            `(distribution, params, transforms)` where:
            
            - `distribution` is a
            `distpy.distribution.Distribution.Distribution` object
            - `params` is a list of the string names of the parameters which
            `distribution` describes
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
                        "provided to the initializer of a DistributionSet " +\
                        "was not a sequence of length 3 like " +\
                        "(distribution, params, transforms).")
        else:
            raise ValueError("The distribution_tuples argument given to " +\
                "the initializer was not list-like. It should be a list of " +\
                "tuples of the form (distribution, params, " +\
                "transformations) where distribution is a Distribution " +\
                "object and params and transformations are lists of strings.")

    @property
    def empty(self):
        """
        Boolean describing whether this `DistributionSet` is empty.
        """
        return (len(self._data) == 0)

    @property
    def params(self):
        """
        The list of string names of the parameters which this `DistributionSet`
        describes.
        """
        if not hasattr(self, '_params'):
            self._params = []
        return self._params
    
    @property
    def discrete_params(self):
        """
        The list of string names of the discrete parameters which this
        `DistributionSet` describes.
        """
        if not hasattr(self, '_discrete_params'):
            self._discrete_params = []
        return self._discrete_params
    
    @property
    def continuous_params(self):
        """
        The list of string names of the continuous parameters which this
        `DistributionSet` describes.
        """
        if not hasattr(self, '_continuous_params'):
            self._continuous_params = []
        return self._continuous_params

    @property
    def numparams(self):
        """
        The total number of parameters described by in this `DistributionSet`.
        """
        return len(self.params)
    
    @property
    def mean(self):
        """
        The approximate mean of this distribution. If the transform has a large
        second derivative at the mean, then this approximation is poor.
        """
        if not hasattr(self, '_mean'):
            mean = {}
            for (distribution, params, transforms) in self._data:
                if distribution.numparams == 1:
                    mean[params[0]] =\
                        transforms[0].apply_inverse(distribution.mean)
                else:
                    this_mean = distribution.mean
                    for (itransform, (parameter, transform)) in\
                        enumerate(zip(params, transforms)):
                        mean[parameter] =\
                            transform.apply_inverse(this_mean[itransform])
            self._mean = mean
        return self._mean
    
    @property
    def variance(self):
        """
        The variances of this distribution in a dictionary indexed by parameter
        name.
        """
        if not hasattr(self, '_variance'):
            variances = {}
            for (distribution, params, transforms) in self._data:
                if distribution.numparams == 1:
                    this_mean = np.array([distribution.mean])
                    this_covariance = np.array([[distribution.variance]])
                    variances[params[0]] =\
                        transforms.inverse.transform_covariance(\
                        this_covariance, this_mean)[0,0]
                else:
                    this_mean = distribution.mean
                    this_covariance = distribution.variance
                    this_untransformed_covariance =\
                        transforms.inverse.transform_covariance(\
                        this_covariance, this_mean)
                    for (parameter, variance) in\
                        zip(params, np.diag(this_untransformed_covariance)):
                        variances[parameter] = variance
            self._variance = variances
        return self._variance
    
    def __len__(self):
        """
        Implemented so that `len(distribution_set)` can be used to get the
        number of parameters of a `DistributionSet` object without explicitly
        referencing `DistributionSet.numparams`.
        
        Returns
        -------
        length : int
            the number of parameter described by this `DistributionSet`
        """
        return self.numparams

    def add_distribution(self, distribution, params, transforms=None):
        """
        Adds a `distpy.distribution.Distribution.Distribution` and the
        parameters it describes to the `DistributionSet`.
        
        Parameters
        ----------
        distribution : `distpy.distribution.Distribution.Distribution`
            the distribution to add
        params : sequence or str
            list of string names of parameters described `distribution` (can be
            a single string if `distribution` is univariate)
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
            if distribution.numparams == 1:
                if type(params) is str:
                    self._check_name(params)
                    self._data.append((distribution, [params], transforms))
                elif type(params) in sequence_types:
                    if len(params) > 1:
                        raise ValueError("The distribution given to a " +\
                            "DistributionSet was univariate, but more than " +\
                            "one parameter was given.")
                    else:
                        self._check_name(params[0])
                        self._data.append(\
                            (distribution, [params[0]], transforms))
                else:
                    raise ValueError("The type of the parameters given " +\
                        "to a DistributionSet was not recognized.")
            elif isinstance(params, basestring):
                raise ValueError(("A single parameter was given even " +\
                   "though the distribution given is multivariate " +\
                   "(numparams={}).").format(distribution.numparams))
            elif type(params) in sequence_types:
                if (len(params) == distribution.numparams):
                    for name in params:
                        self._check_name(name)
                    data_tup = (distribution,\
                        [params[i] for i in range(len(params))], transforms)
                    self._data.append(data_tup)
                else:
                    raise ValueError(("The number of parameters of the " +\
                        "given distribution ({0:d}) was not equal to the " +\
                        "number of parameters given ({1:d}).").format(\
                        distribution.numparams, len(params)))
            else:
                raise ValueError("The params given to a DistributionSet " +\
                    "(along with a distribution) was not a string nor a " +\
                    "list of strings.")
        else:
            raise ValueError("The distribution given to a DistributionSet " +\
                "was not recognized as a distribution.")
        last_distribution_tuple = self._data[-1]
        params_of_last_distribution_tuple = last_distribution_tuple[1]
        self.params.extend(params_of_last_distribution_tuple)
        if distribution.is_discrete:
            self.discrete_params.extend(params_of_last_distribution_tuple)
        else:
            self.continuous_params.extend(params_of_last_distribution_tuple)
    
    def __add__(self, other):
        """
        Adds this `DistributionSet` to another to create a combined set. For
        this to be valid the two `DistributionSet` objects must describe
        different parameters.
        
        Parameters
        ----------
        other : `DistributionSet`
            another `DistributionSet` with parameters distinct from this one
        
        Returns
        -------
        sum : `DistributionSet`
            the combination of the two `DistributionSet` objects being added
        """
        if isinstance(other, DistributionSet):
            if set(self.params) & set(other.params):
                raise ValueError("The two DistributionSet objects shared " +\
                    "at least one parameter.")
            else:
                return\
                    DistributionSet(distribution_tuples=self._data+other._data)
        else:
            return NotImplemented
    
    def __iadd__(self, other):
        """
        Adds all distributions from `other` to this `DistributionSet`.
        
        Parameters
        ----------
        other : `DistributionSet`
            set of distributions to add into this one
        
        Returns
        -------
        enlarged : `DistributionSet`
            this `DistributionSet` after the distributions from `other` have
            been added in
        """
        if isinstance(other, DistributionSet):
            for distribution_tuple in other._data:
                self.add_distribution(*distribution_tuple)
            return self
        else:
            raise TypeError("DistributionSet objects can only have other " +\
                "DistributionSet objects added to them.")
    
    def distribution_list(self, parameters):
        """
        Creates a `distpy.distribution.DistributionList.DistributionList` out
        of this `DistributionSet` by ordering it in the same way as the given
        `parameters`.
        
        Parameters
        ----------
        parameters : sequence
            a sequence of the parameters whose distribution should be put into
            the list, including order. May oy may not contain all of this
            `DistributionSet` object's parameters
        
        Returns
        -------
        list: `distpy.distribution.DistributionList.DistributionList`
            a `distpy.distribution.DistributionList.DistributionList`
            containing the distributions of the given parameters
        """
        to_list = [parameter for parameter in parameters]
        distribution_order = []
        while to_list:
            first_parameter = to_list[0]
            broken = False
            for (ituple, (distribution, params, transforms)) in\
                enumerate(self._data):
                if (first_parameter in params):
                    if ituple in distribution_order:
                        raise ValueError("The same distribution cannot be " +\
                            "put in the same DistributionList twice using " +\
                            "this function. The parameters must be out of " +\
                            "order.")
                    distribution_order.append(ituple)
                    broken = True
                    break
            if not broken:
                raise ValueError(("The parameter {!s} was not found in " +\
                    "this DistributionSet, so couldn't be used to populate " +\
                    "a DistributionList object.").format(first_parameter))
            (distribution, params, transforms) =\
                self._data[distribution_order[-1]]
            for (distribution_param, list_param) in\
                zip(params, to_list[:distribution.numparams]):
                if distribution_param != list_param:
                    raise ValueError("Something went wrong. You must have " +\
                        "parameters out of order with respect to the " +\
                        "distributions they are in.")
            to_list = to_list[distribution.numparams:]
        distribution_tuples =\
            [self._data[element][::2] for element in distribution_order]
        return DistributionList(distribution_tuples=distribution_tuples)
    
    def modify_parameter_names(self, function):
        """
        Modifies the names of the parameters in this distribution by applying
        the given function to each one.
        
        Parameters
        ----------
        function : callable
            a function that takes old names in and outputs new names
        """
        self._params = list(map(function, self.params))
        self._discrete_params = list(map(function, self.discrete_params))
        self._continuous_params = list(map(function, self.continuous_params))
        self._data = [(distribution, list(map(function, params)), transforms)\
            for (distribution, params, transforms) in self._data]
    
    def modify_transforms(self, **new_transforms):
        """
        Creates a `DistributionSet` with the same distribution and parameters
        but different transforms. Draws from this `DistributionSet` and the
        returned `DistributionSet` will differ by the given transforms.
        
        Parameters
        ----------
        new_transforms : dict
            keyword arguments containing some or all parameters of this
            `DistributionSet` as keys and new
            `distpy.transform.Transform.Transform` objects (or things which can
            be cast to them using
            `distpy.transform.CastTransform.cast_to_transform`) as values
        
        Returns
        -------
        modified : `DistributionSet`
            new `DistributionSet` object with the same distribution and
            parameters but different transforms
        """
        new_data = []
        for (distribution, params, transforms) in self._data:
            these_new_transforms = []
            for (iparam, param) in enumerate(params):
                if param in new_transforms:
                    these_new_transforms.append(new_transforms[param])
                else:
                    these_new_transforms.append(transforms[iparam])
            these_new_transforms = TransformList(*these_new_transforms)
            new_data.append((distribution, params, these_new_transforms))
        return DistributionSet(distribution_tuples=new_data)

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
        drawn_points : dict
            a dictionary whose keys are string parameter names and whose values
            are random variates:
            
            - if `shape` is None, then each parameter's random variates are
            floats
            - if `shape` is an integer, then each parameter's random variates
            are 1D arrays of length `shape`
            - if `shape` is a tuple, then each parameter's random variates are
            `len(shape)`-dimensional arrays of shape `shape`
        """
        point = {}
        for (distribution, params, transforms) in self._data:
            if (distribution.numparams == 1):
                (param, transform) = (params[0], transforms[0])
                point[param] = transform.apply_inverse(\
                    distribution.draw(shape=shape, random=random))
            else:
                this_draw = distribution.draw(shape=shape, random=random)
                for iparam in range(len(params)):
                    point[params[iparam]] = transforms[iparam].apply_inverse(\
                        this_draw[...,iparam])
        return point

    def log_value(self, point):
        """
        Evaluates the log of the product of the values of the
        `distpy.distribution.Distribution.Distribution` objects contained in
        this `DistributionSet`, which is the sum of their log values.
        
        Parameters
        ----------
        point : dict
            a dictionary of parameters values indexed by their string names
        
        Returns
        -------
        total_log_value : float
            total log_value coming from contributions from all distributions
        """
        if isinstance(point, dict):
            result = 0.
            for (distribution, params, transforms) in self._data:
                if (distribution.numparams == 1):
                    subpoint = transforms[0].apply(point[params[0]])
                    result += distribution.log_value(subpoint)
                else:
                    subpoint = [transform.apply(point[param])\
                        for (param, transform) in zip(params, transforms)]
                    result += distribution.log_value(subpoint)
                for (param, transform) in zip(params, transforms):
                    if isinstance(transform, NullTransform):
                        continue
                    result += transform.log_derivative(point[param])
            return result
        else:
            raise ValueError("point given to log_value function of a " +\
                "DistributionSet was not a dictionary of values indexed by " +\
                "parameter names.")

    def find_distribution(self, parameter):
        """
        Finds the distribution associated with the given parameter. Also finds
        the index of the parameter in that distribution and the
        `distpy.transform.Transform.Transform` applied to the parameter.
        
        Parameters
        ----------
        parameter : str
            string name of parameter to search for
        
        Returns
        -------
        distribution : `distpy.distribution.Distribution.Distribution`
            the distribution that applies to the given parameter
        index : int
            the integer index describing which parameter of `distribution` is
            the one that was searched for
        transform : `distpy.transform.Transform.Transform`
            the transformation that describes the space that the searched-for
            parameter is drawn in
        """
        for (distribution, params, transforms) in self._data:
            for (iparam, param) in enumerate(params):
                if parameter == param:
                    return (distribution, iparam, transforms[iparam])
        raise ValueError(("The parameter searched for ({!s}) in a " +\
            "DistributionSet was not found.").format(parameter))
    
    def __getitem__(self, parameter):
        """
        Finds the distribution associated with the given parameter. Also finds
        the index of the parameter in that distribution and the
        `distpy.transform.Transform.Transform` applied to the parameter. Alias
        for `DistributionSet.find_distribution` that allows for square bracket
        indexing notation.
        
        Parameters
        ----------
        parameter : str
            string name of parameter to search for
        
        Returns
        -------
        distribution : `distpy.distribution.Distribution.Distribution`
            the distribution that applies to the given parameter
        index : int
            the integer index describing which parameter of `distribution` is
            the one that was searched for
        transform : `distpy.transform.Transform.Transform`
            the transformation that describes the space that the searched-for
            parameter is drawn in
        """
        return self.find_distribution(parameter)

    def delete_distribution(self, parameter, throw_error=True):
        """
        Deletes a distribution from this `DistributionSet`.
        
        Parameters
        ----------
        parameter : str
            any parameter in the distribution to remove
        throw_error : bool
            - if True, then an error is thrown if `parameter` is not the name
            of a parameter of this `DistributionSet`
            - if False, no error is thrown and this method simply returns
            without doing anything if `parameter` is not the name of a
            parameter of this `DistributionSet`
        """
        for (idistribution, distribution_tuple) in enumerate(self._data):
            (distribution, params, transforms) = distribution_tuple
            if parameter in params:
                to_delete = idistribution
                break
        try:
            distribution_to_delete = self._data[to_delete][0]
            is_discrete = distribution_to_delete.is_discrete
            for par in self._data[to_delete][1]:
                self.params.remove(par)
                if is_discrete:
                    self.discrete_params.remove(par)
                else:
                    self.continuous_params.remove(par)
            self._data = self._data[:to_delete] + self._data[to_delete+1:]
        except:
            if throw_error:
                raise ValueError('The parameter given to ' +\
                    'DistributionSet.delete_distribution was not in the ' +\
                    'DistributionSet.')
    
    def __delitem__(self, parameter):
        """
        Deletes a distribution from this `DistributionSet`. Alias for
        `DistributionSet.delete_distribution` with `throw_error` set to True,
        allowing for the `del` keyword to be used to remove distributions.
        
        Parameters
        ----------
        parameter : str
            any parameter in the distribution to remove
        """
        self.delete_distribution(parameter, throw_error=True)
    
    def parameter_strings(self, parameter):
        """
        Makes an informative string about the given parameter's place in this
        `DistributionSet`.
        
        Parameters
        ----------
        parameter : str
            name of parameter being queried
        
        Returns
        -------
        param_string : str
            string describing distribution of parameter
        transform_string : str
            string describing transformation of parameter
        """
        string = ""
        (distribution, index, transform) = self.find_distribution(parameter)
        if distribution.numparams != 1:
            string += (self._numerical_adjective(index) + ' param of ')
        string += distribution.to_string()
        return (string, transform.to_string())
    
    @property
    def summary_string(self):
        """
        A string that summarizes the place of all parameters in this
        `DistributionSet`, including the distributions they belong to and the
        way they are transformed.
        """
        final_string = 'Parameter: distribution   transform'
        for parameter in self.params:
            (distribution_string, transform_string) =\
                self.parameter_strings(parameter)
            final_string = '{0!s}\n{1!s}: {2!s}  {3!s}'.format(final_string,\
                parameter, distribution_string, transform_string)
        return final_string
    
    @staticmethod
    def _distribution_tuples_equal(first, second):
        """
        Checks whether two distribution tuples are equal.
        
        Parameters
        ----------
        first : tuple
            tuple of form `(distribution, parameters, transforms)` as
            internally represented in a `DistributionSet`
        second : tuple
            tuple of form `(distribution, parameters, transforms)` as
            internally represented in a `DistributionSet`
        
        Returns
        -------
        result : bool
            True if and only if the distribution, parameters, and
            transformations stored in `first` are the same as those stored in
            `second`.
        """
        (first_distribution, first_params, first_transforms) = first
        (second_distribution, second_params, second_transforms) = second
        numparams = first_distribution.numparams
        if second_distribution.numparams == numparams:
            for iparam in range(numparams):
                if first_params[iparam] != second_params[iparam]:
                    return False
                if first_transforms[iparam] != second_transforms[iparam]:
                    return False
            return (first_distribution == second_distribution)
        else:
            return False
    
    def __eq__(self, other):
        """
        Checks for equality of this `DistributionSet` with `other`.
        
        Parameters
        ----------
        other : object
            object to check for equality
        
        Returns
        -------
        result : bool
            True if and only if `other` is a `DistributionSet` with the same
            distribution tuples (though they need not be internally stored in
            the same order).
        """
        if isinstance(other, DistributionSet):
            numtuples = len(self._data)
            if len(other._data) == numtuples:
                for idistribution_tuple in range(numtuples):
                    match = False
                    distribution_tuple = self._data[idistribution_tuple]
                    for other_distribution_tuple in other._data:
                        if DistributionSet._distribution_tuples_equal(\
                            distribution_tuple, other_distribution_tuple):
                            match = True
                            break
                    if not match:
                        return False
                return True
            else:
                return False        
        else:
            return False
    
    def __ne__(self, other):
        """
        Checks for inequality of this `DistributionSet` with `other`.
        
        Parameters
        ----------
        other : object
            object to check for inequality
        
        Returns
        -------
        result : bool
            False if and only if `other` is a `DistributionSet` with the same
            distribution tuples (though they need not be internally stored in
            the same order).
        """
        return (not self.__eq__(other))

    def _numerical_adjective(self, num):
        """
        Creates a numerical adjective, such as '1st', '2nd', '6th' and so on.
        
        Parameters
        ----------
        num : int
            an integer for which to make an adjective form
        
        Returns
        -------
        adjective : str
            string adjective describing name
        """
        if (type(num) in int_types) and (num >= 0):
            base_string = str(num)
            if num == 0:
                return '0th'
            elif num == 1:
                return '1st'
            elif num == 2:
                return '2nd'
            elif num == 3:
                return '3rd'
            else:
                return str(num) + 'th'
        else:
            raise ValueError("Numerical adjectives apply only to " +\
                "non-negative integers.")

    def _check_name(self, name):
        """
        Checks the given name to see if it is already taken in the parameters
        of the distributions in this `DistributionSet` and raises a
        `ValueError` if it is.
        
        Parameters
        ----------
        name : str
            string parameter name to check for
        """
        if not isinstance(name, basestring):
            raise ValueError("A parameter provided to a DistributionSet " +\
                "was not a string.")
        broken = False
        for (distribution, params, transforms) in self._data:
            if name in params:
                broken = True
                break
        if broken:
            raise ValueError("The name of a parameter provided to a " +\
                "DistributionSet is already taken.")
    
    @property
    def transform_set(self):
        """
        A `distpy.transform.TransformSet.TransformSet` object describing the
        transforms in this `DistributionSet` and the parameters they apply to.
        """
        transforms_dictionary = {}
        for (distribution, params, transforms) in self._data:
            for (param, transform) in zip(params, transforms):
                transforms_dictionary[param] = transform
        return TransformSet(transforms_dictionary)
    
    def discrete_subset(self):
        """
        Compiles the subset of the `Distribution` objects in this
        `DistributionSet` that represent discrete variables.
        
        Returns
        -------
        subset : `DistributionSet`
            a `DistributionSet` object containing all
            `distpy.distribution.Distribution.Distribution` objects in this
            `DistributionSet` which describe discrete variables
        """
        answer = DistributionSet()
        for (distribution, params, transforms) in self._data:
            if distribution.is_discrete:
                answer.add_distribution(distribution, params, transforms)
        return answer
    
    def continuous_subset(self):
        """
        Compiles the subset of the `Distribution` objects in this
        `DistributionSet` that represent continuous variables.
        
        Returns
        -------
        subset : `DistributionSet`
            a `DistributionSet` object containing all
            `distpy.distribution.Distribution.Distribution` objects in this
            `DistributionSet` which describe continuous variables
        """
        answer = DistributionSet()
        for (distribution, params, transforms) in self._data:
            if not distribution.is_discrete:
                answer.add_distribution(distribution, params, transforms)
        return answer
    
    def transformed_version(self):
        """
        Compiles a version of this `DistributionSet` where the parameters exist
        in transformed space (instead of transforms being carried through this
        object).
        
        Returns
        -------
        transformless : `DistributionSet`
            a `DistributionSet` with the same distributions and parameter names
            but without transforms
        """
        answer = DistributionSet()
        for (distribution, params, transforms) in self._data:
            answer.add_distribution(distribution, params)
        return answer
    
    def fill_hdf5_group(self, group, save_metadata=True):
        """
        Fills the given hdf5 file group with data about this `DistributionSet`.
        
        Parameters
        ----------
        group : h5py.Group
            the hdf5 file group to fill
        save_metadata : bool
            - if True, attempts to save metadata alongside distribution set and
            throws error if it fails
            - if False, metadata is ignored in saving process
        """
        for (ituple, distribution_tuple) in enumerate(self._data):
            (distribution, params, transforms) = distribution_tuple
            subgroup = group.create_group('distribution_{}'.format(ituple))
            distribution.fill_hdf5_group(subgroup, save_metadata=save_metadata)
            transforms.fill_hdf5_group(subgroup)
            for (iparam, param) in enumerate(params):
                subgroup.attrs['parameter_{}'.format(iparam)] = param
    
    @property
    def minimum(self):
        """
        The minimum allowable value(s) of the parameters in a dictionary.
        """
        if not hasattr(self, '_minimum'):
            self._minimum = {}
            for (distribution, params, transforms) in self._data:
                if distribution.numparams == 1:
                    self._minimum[params[0]] =\
                        transforms[0].untransform_minimum(distribution.minimum)
                else:
                    these_minima = list(distribution.minimum)
                    for (iparam, (param, transform)) in\
                        enumerate(zip(params, transforms)):
                        self._minimum[param] =\
                            transform.untransform_minimum(these_minima[iparam])
        return self._minimum
    
    @property
    def maximum(self):
        """
        The maximum allowable value(s) of the parameters in a dictionary.
        """
        if not hasattr(self, '_maximum'):
            self._maximum = {}
            for (distribution, params, transforms) in self._data:
                if distribution.numparams == 1:
                    self._maximum[params[0]] =\
                        transforms[0].untransform_maximum(distribution.maximum)
                else:
                    these_maxima = list(distribution.maximum)
                    for (iparam, (param, transform)) in\
                        enumerate(zip(params, transforms)):
                        self._maximum[param] =\
                            transform.untransform_maximum(these_maxima[iparam])
        return self._maximum
    
    @property
    def bounds(self):
        """
        The minimum and maximum allowable value(s) of the parameters in a
        dictionary.
        """
        if not hasattr(self, '_bounds'):
            self._bounds = {}
            for parameter in self.params:
                self._bounds[parameter] =\
                    (self.minimum[parameter], self.maximum[parameter])
        return self._bounds
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a `DistributionSet` object from the given hdf5 file group.
        
        Parameters
        ----------
        group : h5py.Group
            the group which a `DistributionSet` was once saved in
        
        Returns
        -------
        loaded : `DistributionSet`
            the loaded `DistributionSet`
        """
        ituple = 0
        distribution_tuples = []
        while ('distribution_{}'.format(ituple)) in group:
            subgroup = group['distribution_{}'.format(ituple)]
            distribution = load_distribution_from_hdf5_group(subgroup)
            transform_list = TransformList.load_from_hdf5_group(subgroup)
            params = []
            iparam = 0
            for iparam in range(distribution.numparams):
                params.append(subgroup.attrs['parameter_{}'.format(iparam)])
            distribution_tuples.append((distribution, params, transform_list))
            ituple += 1
        return DistributionSet(distribution_tuples=distribution_tuples)
    
    @property
    def gradient_computable(self):
        """
        Boolean describing whether the gradient of the distributions inside
        this `DistributionSet` have been implemented.
        """
        answer = True
        for (distribution, params, transforms) in self._data:
            answer = (answer and distribution.gradient_computable)
        return answer
    
    def gradient_of_log_value(self, point):
        """
        Computes the derivatives of the log value with respect to the
        parameters.
        
        Parameters
        ----------
        point : dict
            a dictionary of parameters values indexed by their string names
        
        Returns
        -------
        gradient : `numpy.ndarray`
            1D array of length `DistributionSet.numparams` of derivative values
            corresponding to the parameters (in the order given by
            `DistributionSet.params`)
        """
        if isinstance(point, dict):
            result = np.zeros((self.numparams,))
            iparam = 0
            for (idistribution, distribution_tuple) in enumerate(self._data):
                (distribution, params, transforms) = distribution_tuple
                next_iparam = iparam + len(params)
                if (distribution.numparams == 1):
                    result[iparam] += distribution.gradient_of_log_value(\
                        transforms[0].apply(point[params[0]]))
                else:
                    subpoint = [transform.apply(point[param])\
                        for (param, transform) in zip(params, transforms)]
                    result[iparam:next_iparam] +=\
                        distribution.gradient_of_log_value(subpoint)
                for i in range(len(params)):
                    result[iparam+i] +=\
                        transforms[i].derivative_of_log_derivative(\
                        point[params[i]])
                iparam = next_iparam
            return result
        else:
            raise ValueError("point given to gradient_of_log_value " +\
                "function of a DistributionSet was not a dictionary of " +\
                "values indexed by parameter names.")
    
    @property
    def hessian_computable(self):
        """
        Boolean describing whether the hessian of the distributions inside
        this `DistributionSet` have been implemented.
        """
        answer = True
        for (distribution, params, transforms) in self._data:
            answer = (answer and distribution.hessian_computable)
        return answer
    
    def hessian_of_log_value(self, point):
        """
        Computes the second derivatives of the log value with respect to the
        parameters.
        
        Parameters
        ----------
        point : dict
            a dictionary of parameters values indexed by their string names
        
        Returns
        -------
        gradient : `numpy.ndarray`
            square 2D array of length `DistributionSet.numparams` of second
            derivative values corresponding to the parameters (in the order
            given by `DistributionSet.params`)
        """
        if isinstance(point, dict):
            result = np.zeros((self.numparams,) * 2)
            iparam = 0
            for (idistribution, distribution_tuple) in enumerate(self._data):
                (distribution, params, transforms) = distribution_tuple
                next_iparam = iparam + len(params)
                if (distribution.numparams == 1):
                    subpoint = transforms[0].apply(point[params[0]])
                    result[iparam,iparam] +=\
                        distribution.hessian_of_log_value(subpoint)
                else:
                    subpoint = [transform.apply(point[param])\
                        for (param, transform) in zip(params, transforms)]
                    result[iparam:next_iparam,iparam:next_iparam] +=\
                        distribution.hessian_of_log_value(subpoint)
                for i in range(len(params)):
                    result[iparam+i,iparam+i] +=\
                        transforms[i].second_derivative_of_log_derivative(\
                        point[params[i]])
                iparam = next_iparam
            return result
        else:
            raise ValueError("point given to hessian_of_log_value " +\
                "function of a DistributionSet was not a dictionary of " +\
                "values indexed by parameter names.")
    
    def copy(self):
        """
        Finds a deep copy of this `DistributionSet`.
        
        Returns
        -------
        copied : `DistributionSet`
            deep copy of this `DistributionSet`
        """
        copied = DistributionSet()
        for (distribution, params, transforms) in self._data:
            copied_distribution = distribution.copy()
            copied_params = [param for param in params]
            copied_transforms = [transform for transform in transforms]
            copied.add_distribution(\
                copied_distribution, copied_params, copied_transforms)
        return copied
    
    def reset(self):
        """
        Resets this distribution. This allows ideal distributions to live
        alongside samples as the same kind of object.
        """
        for (distribution, parameters, transforms) in self._data:
            distribution.reset()
    
    def plot_univariate_histogram(self, ndraw, parameter,\
        in_transformed_space=True, reference_value=None, bins=None,\
        matplotlib_function='fill_between', show_intervals=False,\
        norm_by_max=True, xlabel='', ylabel='', title='', fontsize=28,\
        ax=None, show=False, **kwargs):
        """
        Plots a histogram of drawn values of the given parameter.
        
        Parameters
        ----------
        ndraw : int
            the number of points to draw from the distribution
        parameter : str
            the string name of the parameter to plot
        in_transformed_space : bool
            boolean determining whether the points drawn from inner
            distribution are plotted directly (True) or are untransformed
            (False)
        reference_value : real number or None
            if given, a point at which to plot a dashed reference line
        bins : int, sequence, or None
            bins to pass to `numpy.histogram` function
        matplotlib_function : str
            either 'fill_between', 'bar', or 'plot'
        show_intervals : bool
            if True, confidence intervals are plotted
        norm_by_max : bool
            if True, normalization is such that maximum of histogram values is
            1.
        xlabel : str
            the string to use in labeling x axis
        ylabel : str
            the string to use in labeling y axis
        title : str
            title string with which to top plot
        fontsize : int, str, or None
            integer size in points or one of ['xx-small', 'x-small', 'small',
            'medium', 'large', 'x-large', 'xx-large'] representing size of
            labels
        ax : matplotlib.Axes or None
            - if None, new Figure and Axes are created  
            - otherwise, this Axes object is plotted on
        show : bool
            if True, `matplotlib.pyplot.show` is called before this function
            returns
        kwargs : dict
            keyword arguments to pass on to `matplotlib.Axes.plot` or
            `matplotlib.Axes.fill_between`
        
        Returns
        -------
        axes : `matplotlib.Axes` or None
            - if `show` is True, `axes` is None
            - if `show` is False, `axes` is the `matplotlib.Axes` instance with
            plot
        """
        sample = self.draw(ndraw)
        if in_transformed_space:
            sample = self.transform_set[parameter](sample[parameter])
        else:
            sample = sample[parameter]
        return univariate_histogram(sample, reference_value=reference_value,\
            bins=bins, matplotlib_function=matplotlib_function,\
            show_intervals=show_intervals, xlabel=xlabel, ylabel=ylabel,\
            title=title, fontsize=fontsize, ax=ax, show=show,\
            norm_by_max=norm_by_max, **kwargs)
    
    def plot_bivariate_histogram(self, ndraw, parameter1, parameter2,\
        in_transformed_space=True, reference_value_mean=None,\
        reference_value_covariance=None, bins=None,\
        matplotlib_function='imshow', xlabel='', ylabel='', title='',\
        fontsize=28, ax=None, show=False, contour_confidence_levels=0.95,\
        **kwargs):
        """
        Plots a 2D histogram of the given joint sample.
        
        Parameters
        ----------
        ndraw : int
            the number of points to draw from the distribution
        parameter1 : str
            the string name of the x-parameter to plot
        parameter2 : str
            the string name of the y-parameter to plot
        in_transformed_space : bool
            boolean determining whether the points drawn from inner
            distribution are plotted directly (True) or are untransformed
            (False)
        reference_value_mean : sequence or None
            - if None, no reference line is plotted  
            - otherwise, sequence of two elements representing the reference
            value for x- and y-samples. Each element can be either None (if no
            reference line should be plotted) or a value at which to plot a
            reference line.
        reference_value_covariance: numpy.ndarray or None
            - if `numpy.ndarray`, represents the covariance matrix used to
            generate a reference ellipse around the reference mean.  
            - if None or if one or more of `reference_value_mean` is None, no
            ellipse is plotted
        bins : int, sequence, or None
            bins to pass to `numpy.histogram2d`
        matplotlib_function : str
            function to use in plotting. One of ['imshow', 'contour',
            'contourf'].
        xlabel : str
            the string to use in labeling x axis
        ylabel : str
            the string to use in labeling y axis
        title : str
            title string with which to top plot
        fontsize : int, str, or None
            integer size in points or one of ['xx-small', 'x-small', 'small',
            'medium', 'large', 'x-large', 'xx-large'] representing size of
            labels
        ax : matplotlib.Axes or None
            - if None, new Figure and Axes are created  
            - otherwise, this Axes object is plotted on
        show : bool
            if True, `matplotlib.pyplot.show` is called before this function
            returns
        contour_confidence_levels : number or sequence of numbers
            confidence level as a number between 0 and 1 or a 1D array of such
            numbers. Only used if `matplotlib_function` is `'contour'` or
            `'contourf'` or if `reference_value_mean` and
            `reference_value_covariance` are both not None
        kwargs : dict
            keyword arguments to pass on to `matplotlib.Axes.imshow` (any but
            'origin', 'extent', or 'aspect') or `matplotlib.Axes.contour` or
            `matplotlib.Axes.contourf` (any)
        
        Returns
        -------
        axes : `matplotlib.Axes` or None
            - if `show` is True, `axes` is None
            - if `show` is False, `axes` is the `matplotlib.Axes` instance with
            plot
        """
        samples = self.draw(ndraw)
        parameters = [parameter1, parameter2]
        minima = {parameter: ((-np.inf) if (type(self.minimum[parameter]) is\
            type(None)) else self.minimum[parameter])\
            for parameter in parameters}
        maxima = {parameter: ((+np.inf) if (type(self.maximum[parameter]) is\
            type(None)) else self.maximum[parameter])\
            for parameter in parameters}
        if in_transformed_space:
            samples = [self.transform_set[parameter](samples[parameter])\
                for parameter in parameters]
            (minima, maxima) = ([], [])
            for parameter in parameters:
                if type(self.minimum[parameter]) is type(None):
                    minima.append(self.transform_set[parameter](-np.inf))
                else:
                    minima.append(\
                        self.transform_set[parameter](self.minimum[parameter]))
                if type(self.maximum[parameter]) is type(None):
                    maxima.append(self.transform_set[parameter](+np.inf))
                else:
                    maxima.append(\
                        self.transform_set[parameter](self.maximum[parameter]))
        else:
            samples = [samples[parameter] for parameter in parameters]
            (minima, maxima) = ([], [])
            for parameter in parameters:
                if type(self.minimum[parameter]) is type(None):
                    minima.append(-np.inf)
                else:
                    minima.append(self.minimum[parameter])
                if type(self.maximum[parameter]) is type(None):
                    maxima.append(+np.inf)
                else:
                    maxima.append(self.maximum[parameter])
        return bivariate_histogram(samples[0], samples[1],\
            reference_value_mean=reference_value_mean,\
            reference_value_covariance=reference_value_covariance, bins=bins,\
            matplotlib_function=matplotlib_function, xlabel=xlabel,\
            ylabel=ylabel, title=title, fontsize=fontsize, ax=ax, show=show,\
            contour_confidence_levels=contour_confidence_levels,\
            minima=minima, maxima=maxima, **kwargs)
    
    def triangle_plot(self, ndraw, parameters=None, in_transformed_space=True,\
        figsize=(8, 8), fig=None, show=False, kwargs_1D={}, kwargs_2D={},\
        fontsize=28, nbins=100, plot_type='contour', plot_limits=None,\
        reference_value_mean=None, reference_value_covariance=None,\
        contour_confidence_levels=0.95, parameter_renamer=(lambda x: x),\
        tick_label_format_string='{x:.3g}', num_ticks=3,\
        minor_ticks_per_major_tick=1, xlabel_rotation=0, xlabelpad=None,\
        ylabel_rotation=90, ylabelpad=None):
        """
        Makes a triangle plot out of ndraw samples from this distribution
        
        Parameters
        ----------
        ndraw : int
            the number of points to draw from the distribution
        parameters : sequence
            sequence of parameter names to include in the triangle plot
        in_transformed_space : bool
            boolean determining whether the points drawn from inner
            distribution are plotted directly (True) or are untransformed
            (False)
        figsize : tuple
            tuple of form (width, height) representing the size of the figure
            on which to put the triangle plot
        fig : `matplotlib.Figure` or None
            - if provided, `fig` will be plotted on
            - otherwise, a new `matplotlib.Figure` is created
        show : bool
            if True, `matplotlib.pyplot.show` is called before this function
            returns
        kwargs_1D : dict
            keyword arguments to pass on to
            `distpy.util.TrianglePlot.univariate_histogram` function
        kwargs_2D : dict
            keyword arguments to pass on to
            `distpy.util.TrianglePlot.bivariate_histogram` function
        fontsize : int, str, or None
            integer size in points or one of ['xx-small', 'x-small', 'small',
            'medium', 'large', 'x-large', 'xx-large'] representing size of
            labels
        nbins : int
            the number of bins to use for each sample
        plot_type : str or sequence
            determines the matplotlib functions to use for univariate and
            bivariate histograms
        
            - if `plot_type=='contourf'`: 'bar' and 'contourf' are used
            - if `plot_type=='contour'`: 'plot' and 'contour' are used
            - if `plot_type=='histogram'`: 'bar' and 'imshow' are used
            - otherwise: plot_type should be a length-2 sequence of the form
            (matplotlib_function_1D, matplotlib_function_2D)
        plot_limits : sequence or None
            - if None, bins are used to decide plot limits  
            - otherwise, a sequence of 2-tuples of the form (low, high)
            representing the desired axis limits for each variable
        reference_value_mean : sequence or None
            sequence of reference values to place on plots. Each element of the
            sequence (representing each random variable) can be either a number
            at which to plot a reference line or None if no line should be
            plotted. Alternatively, if `reference_value_mean` is set to None,
            no reference lines are plotted for any variable
        reference_value_covariance : numpy.ndarray or None
            covariance with which to create reference ellipses around
            `reference_value_mean`. Should be an NxN array where N is the
            number of random variables. If any of `reference_value_mean` are
            None or `reference_value_covariance` is None, then no ellipses are
            plotted
        contour_confidence_levels : number or sequence of numbers
            confidence level as a number between 0 and 1 or a 1D array of such
            numbers. Only used if `matplotlib_function` is `'contour'` or
            `'contourf'` or if `reference_value_mean` and
            `reference_value_covariance` are both not None
        parameter_renamer : callable
            a function that modifies the parameter names to their label strings
        tick_label_format_string : str
            format string that can be called using
            `tick_label_format_string.format(x=loc)` where `loc` is the
            location of the tick in data coordinates
        num_ticks : int
            number of major ticks in each panel
        minor_ticks_per_major_tick : int
            number of minor ticks per major tick in each panel
        xlabel_rotation : number
            rotation of x-label in degrees
        xlabelpad : number or None
            pad size for xlabel or None if none should be used
        ylabel_rotation : number
            rotation of y-label in degrees
        ylabelpad : number or None
            pad size for ylabel or None if none should be used
        
        Returns
        -------
        figure : matplotlib.Figure or None
            - if `show` is True, None is returned  
            - otherwise, the matplotlib.Figure instance plotted on is returned
        """
        samples = self.draw(ndraw)
        if type(parameters) is type(None):
            parameters = self.params
        minima = {parameter: ((-np.inf) if (type(self.minimum[parameter]) is\
            type(None)) else self.minimum[parameter])\
            for parameter in parameters}
        maxima = {parameter: ((+np.inf) if (type(self.maximum[parameter]) is\
            type(None)) else self.maximum[parameter])\
            for parameter in parameters}
        if in_transformed_space:
            samples = [self.transform_set[parameter](samples[parameter])\
                for parameter in parameters]
            minima = [self.transform_set[parameter](minima[parameter])\
                for parameter in parameters]
            maxima = [self.transform_set[parameter](maxima[parameter])\
                for parameter in parameters]
        else:
            samples = [samples[parameter] for parameter in parameters]
            minima = [self.minimum[parameter] for parameter in parameters]
            maxima = [self.maximum[parameter] for parameter in parameters]
        labels = [parameter_renamer(parameter) for parameter in parameters]
        if type(plot_limits) is not type(None):
            if in_transformed_space:
                plot_limits = [\
                    (self.transform_set[parameter](plot_limits[parameter][0]),\
                    self.transform_set[parameter](plot_limits[parameter][1]))\
                    for parameter in parameters]
            else:
                plot_limits =\
                    [plot_limits[parameter] for parameter in parameters]
        return triangle_plot(samples, labels, figsize=figsize, fig=fig,\
            show=show, kwargs_1D=kwargs_1D, kwargs_2D=kwargs_2D,\
            fontsize=fontsize, nbins=nbins, plot_type=plot_type,\
            reference_value_mean=reference_value_mean,\
            reference_value_covariance=reference_value_covariance,\
            contour_confidence_levels=contour_confidence_levels,\
            minima=minima, maxima=maxima, plot_limits=plot_limits,\
            tick_label_format_string=tick_label_format_string,\
            num_ticks=num_ticks,\
            minor_ticks_per_major_tick=minor_ticks_per_major_tick,\
            xlabel_rotation=xlabel_rotation, xlabelpad=xlabelpad,\
            ylabel_rotation=ylabel_rotation, ylabelpad=ylabelpad)

