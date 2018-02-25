"""
File: distpy/distribution/DistributionSet.py
Author: Keith Tauscher
Date: 12 Feb 2018

Description: A container which can hold an arbitrary number of distributions,
             each of which can have any number of parameters which it describes
             (as long as the specific distribution supports that number of
             parameters). Distribution objects can be added through
             DistributionSet.add_distribution(distribution, params) where
             distribution is a Distribution and params is a list of the
             parameters to which distribution applies. Once all the
             distributions are added, points can be drawn using
             DistributionSet.draw() and the log_value of the entire set of
             distributions can be evaluated at a point using
             DistributionSet.log_value(point). See documentation of individual
             functions for further details.
"""
import numpy as np
from ..util import Savable, Loadable, int_types, sequence_types
from ..transform import cast_to_transform_list, TransformList, TransformSet
from .Distribution import Distribution
from .LoadDistribution import load_distribution_from_hdf5_group
try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str

class DistributionSet(Savable, Loadable):
    """
    An object which keeps track of many distributions which can be univariate
    or multivariate. It provides methods like log_value, which calls log_value
    on all of its constituent distributions, and draw, which draws from all of
    its constituent distributions.
    """
    def __init__(self, distribution_tuples=[]):
        """
        Creates a new DistributionSet with the given distributions inside.
        
        distribution_tuples: a list of lists/tuples of the form
                             (distribution, params) where distribution is an
                             instance of the Distribution class and params is a
                             list of parameters (strings) which distribution
                             describes
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
        Finds whether this DistributionSet is empty.
        
        returns True if no distributions have been added, False otherwise
        """
        return (len(self._data) == 0)

    @property
    def params(self):
        """
        Finds and returns the parameters which this DistributionSet describes.
        """
        if not hasattr(self, '_params'):
            self._params = []
        return self._params
    
    @property
    def discrete_params(self):
        """
        Finds and returns the discrete parameters which this DistributionSet
        describes.
        """
        if not hasattr(self, '_discrete_params'):
            self._discrete_params = []
        return self._discrete_params
    
    @property
    def continuous_params(self):
        """
        Finds and returns the continuous parameters which this DistributionSet
        describes.
        """
        if not hasattr(self, '_continuous_params'):
            self._continuous_params = []
        return self._continuous_params

    @property
    def numparams(self):
        """
        Property storing the number of parameters in this DistributionSet.
        """
        return len(self.params)

    def add_distribution(self, distribution, params, transforms=None):
        """
        Adds a distribution and the parameters it describes to the
        DistributionSet.
        
        distribution: Distribution object describing the given parameters
        params: list of parameters described by the given distribution
                (can be a single string if the distribution is univariate)
        transforms: TransformList object (or something castable to one, such as
                    a sequence of strings which can be cast to Transform
                    objects) which apply to the parameters (can be a single
                    string if the distribution is univariate)
        """
        if isinstance(distribution, Distribution):
            transforms = cast_to_transform_list(transforms,\
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
        Adds this DistributionSet to another.
        
        other: a DistributionSet object with parameters distinct from the
               parameters of self
        
        returns: DistributionSet object which is the combination of the given
                 DistributionSet objects
        """
        if isinstance(other, DistributionSet):
            if set(self.params) & set(other.params):
                raise ValueError("The two DistributionSet objects shared " +\
                    "at least one parameter.")
            else:
                return\
                    DistributionSet(distribution_tuples=self._data+other._data)
        else:
            raise TypeError("Can only add DistributionSet objects to other " +\
                "DistributionSet objects.")
    
    def __iadd__(self, other):
        """
        Adds all distributions from other to this DistributionSet.
        
        other: DistributionSet object with parameters distinct from the
               parameters of self
        """
        if isinstance(other, DistributionSet):
            for distribution_tuple in other._data:
                self.add_distribution(*distribution_tuple)
            return self
        else:
            raise TypeError("DistributionSet objects can only have other " +\
                "DistributionSet objects added to them.")

    def draw(self, shape=None):
        """
        Draws a point from all distributions.
        
        shape: shape of arrays which are values of return value
        
        returns a dictionary of random values indexed by parameter name
        """
        point = {}
        for (distribution, params, transforms) in self._data:
            if (distribution.numparams == 1):
                (param, transform) = (params[0], transforms[0])
                point[param] = transform.apply_inverse(\
                    distribution.draw(shape=shape))
            else:
                this_draw = distribution.draw(shape=shape)
                for iparam in range(len(params)):
                    point[params[iparam]] = transforms[iparam].apply_inverse(\
                        this_draw[...,iparam])
        return point

    def log_value(self, point):
        """
        Evaluates the log of the product of the values of the distributions
        contained in this DistributionSet.
        
        point: should be a dictionary of values indexed by the parameter names
        
        returns: the total log_value coming from contributions from all
                 distributions
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
                    result += transform.log_derivative(point[param])
            return result
        else:
            raise ValueError("point given to log_value function of a " +\
                "DistributionSet was not a dictionary of values indexed by " +\
                "parameter names.")

    def find_distribution(self, parameter):
        """
        Finds the distribution associated with the given parameter. Also finds
        the index of the parameter in that distribution and the transformation
        applied to the parameter.
        
        parameter string name of parameter
        """
        for (distribution, params, transforms) in self._data:
            for (iparam, param) in enumerate(params):
                if parameter == param:
                    return (distribution, iparam, transforms[iparam])
        raise ValueError(("The parameter searched for ({!s}) in a " +\
            "DistributionSet was not found.").format(parameter))
    
    def __getitem__(self, parameter):
        """
        Returns the same thing as: self.find_distribution(parameter)
        """
        return self.find_distribution(parameter)

    def delete_distribution(self, parameter, throw_error=True):
        """
        Deletes a distribution from this DistributionSet.
        
        parameter: a parameter in the distribution
        throw_error: if True (default), an error is thrown if the parameter
                     is not found
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
        Deletes the distribution associated with the given parameter. For
        documentation, see delete_distribution function.
        """
        self.delete_distribution(parameter, throw_error=True)
    
    def parameter_strings(self, parameter):
        """
        Makes an informative string about this parameter's place in this
        DistributionSet.
        
        parameter string name of parameter
        
        returns (param_string, transform_string) in tuple form
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
        Property which yields a string which summarizes the place of all
        parameters in this DistributionSet, including the distributions they
        belong to and the way they are transformed.
        """
        final_string = 'Parameter: distribution   transform'
        for parameter in self.params:
            (distribution_string, transform_string) =\
                self.parameter_strings(parameter)
            final_string = '{0!s}\n{1!s}: {2!s}  {3!s}'.format(final_string,\
                parameter, distribution_string, transform_string)
        return final_string
    
    def __eq__(self, other):
        """
        Checks for equality of this DistributionSet with other. Returns True if
        otherhas the same distribution_tuples (though they need not be
        internally stored in the same order) and False otherwise.
        """
        def distribution_tuples_equal(first, second):
            #
            # Checks whether two distribution_tuple's are equal. Returns True
            # if the distribution, params, and transforms stored in first are
            # the same as those stored in second and False otherwise.
            #
            fdistribution, fparams, ftfms = first
            sdistribution, sparams, stfms = second
            numparams = fdistribution.numparams
            if sdistribution.numparams == numparams:
                for iparam in range(numparams):
                    if fparams[iparam] != sparams[iparam]:
                        return False
                    if ftfms[iparam] != stfms[iparam]:
                        return False
                return (fdistribution == sdistribution)
            else:
                return False
        if isinstance(other, DistributionSet):
            numtuples = len(self._data)
            if len(other._data) == numtuples:
                for idistribution_tuple in range(numtuples):
                    match = False
                    distribution_tuple = self._data[idistribution_tuple]
                    for other_distribution_tuple in other._data:
                        if distribution_tuples_equal(distribution_tuple,\
                            other_distribution_tuple):
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
        This function simply asserts that (a != b) == (not (a == b))
        """
        return (not self.__eq__(other))

    def _numerical_adjective(self, num):
        #
        # Creates a numerical adjective, such as '1st', '2nd', '6th' and so on.
        #
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
        #
        # Checks the given name to see if it is already taken in the parameters
        # of the distributions in this DistributionSet.
        #
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
        Property storing the TransformSet object describing the transforms in
        this DistributionSet.
        """
        if not hasattr(self, '_transform_set'):
            transforms_dictionary = {}
            for (distribution, params, transforms) in self._data:
                for (param, transform) in zip(params, transforms):
                    transforms_dictionary[param] = transform
            self._transform_set = TransformSet(transforms_dictionary)
        return self._transform_set
    
    def discrete_subset(self):
        """
        Function which compiles a subset of the Distribution objects in this
        DistributionSet: those that represent discrete variables.
        
        returns: a DistributionSet object containing all Distribution objects
                 in this DistributionSet which describe discrete variables
        """
        answer = DistributionSet()
        for (distribution, params, transforms) in self._data:
            if distribution.is_discrete:
                answer.add_distribution((distribution, params, transforms))
        return answer
    
    def continuous_subset(self):
        """
        Function which compiles a subset of the Distribution objects in this
        DistributionSet: those that represent continuous variables.
        
        returns: a DistributionSet object containing all Distribution objects
                 in this DistributionSet which describe continuous variables
        """
        answer = DistributionSet()
        for (distribution, params, transforms) in self._data:
            if not distribution.is_discrete:
                answer.add_distribution((distribution, params, transforms))
        return answer
    
    def fill_hdf5_group(self, group, save_metadata=True):
        """
        Fills the given hdf5 file group with data about this DistributionSet.
        Each distribution tuple is saved as a subgroup in the hdf5 file.
        
        group: the hdf5 file group to fill
        save_metadata: if True, attempts to save metadata alongside
                                distribution set and throws error if it fails
                       if False, metadata is ignored in saving process
        """
        for (ituple, distribution_tuple) in enumerate(self._data):
            (distribution, params, transforms) = distribution_tuple
            subgroup = group.create_group('distribution_{}'.format(ituple))
            distribution.fill_hdf5_group(subgroup, save_metadata=save_metadata)
            transforms.fill_hdf5_group(subgroup)
            for (iparam, param) in enumerate(params):
                subgroup.attrs['parameter_{}'.format(iparam)] = param
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a DistributionSet object from the given group.
        
        group: the group which was included in self.fill_hdf5_group(group)
        
        returns: DistributionSet object
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
        Property which stores whether the gradient of the given distribution
        has been implemented.
        """
        if not hasattr(self, '_gradient_computable'):
            self._gradient_computable = True
            for (distribution, params, transforms) in self._data:
                self._gradient_computable = (self._gradient_computable and\
                    distribution.gradient_computable)
        return self._gradient_computable
    
    def gradient_of_log_value(self, point):
        """
        Computes the derivative(s) of log_value(point) with respect to the
        parameter(s).
        
        point: either single value (if distribution is 1D) or array of values
        
        returns: if distribution is 1D, returns single number representing
                                        derivative of log value
                 else, returns 1D numpy.ndarray containing the N derivatives of
                       the log value with respect to each individual parameter
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
        Property which stores whether the hessian of the given distribution
        has been implemented.
        """
        if not hasattr(self, '_hessian_computable'):
            self._hessian_computable = True
            for (distribution, params, transforms) in self._data:
                self._hessian_computable = (self._hessian_computable and\
                    distribution.hessian_computable)
        return self._hessian_computable
    
    def hessian_of_log_value(self, point):
        """
        Computes the second derivative(s) of log_value(point) with respect to
        the parameter(s).
        
        point: either single value (if distribution is 1D) or array of values
        
        returns: if distribution is 1D, returns single number representing
                                        second derivative of log value
                 else, returns 2D square numpy.ndarray with dimension length
                       equal to the number of parameters representing the N^2
                       different second derivatives of the log value
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

