"""
File: distpy/DistributionSet.py
Author: Keith Tauscher
Date: 6 Aug 2017

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
from ..util import Savable, int_types, sequence_types
from ..transform import NullTransform, cast_to_transform, castable_to_transform
from .Distribution import Distribution        

class DistributionSet(Savable):
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
        self._params = []
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
        return self._params

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
        params list of parameters described by the given distribution
               (can be a single string if the distribution is univariate)
        transforms list of transformations to apply to the parameters
                   (can be a single string if the distribution is univariate)
        """
        if isinstance(distribution, Distribution):
            if transforms is None:
                transforms = [NullTransform()] * distribution.numparams
            elif castable_to_transform(transforms):
                if (distribution.numparams == 1):
                    transforms = [cast_to_transform(transforms)]
                else:
                    raise ValueError("The transforms variable applied to " +\
                        "parameters of a DistributionSet was provided as a " +\
                        "string even though the distribution being " +\
                        "provided was multivariate.")
            elif type(transforms) in sequence_types:
                if len(transforms) == distribution.numparams:
                    all_castable_to_transforms =\
                        all([castable_to_transform(val) for val in transforms])
                    if all_castable_to_transforms:
                        transforms =\
                            [cast_to_transform(val) for val in transforms]
                    else:
                        raise ValueError("Not all transforms given to " +\
                            "add_distribution were understood.")
                else:
                    raise ValueError("The list of transforms applied to " +\
                        "parameters in a DistributionSet was not the same " +\
                        "length as the list of parameters of the " +\
                        "distribution.")
            else:
                raise ValueError("The type of the transforms variable " +\
                    "supplied to DistributionSet's add_distribution " +\
                    "function was not recognized. It should be a single " +\
                    "valid string (if distribution is univariate) or list " +\
                    "of valid strings (if distribution is multivariate).")
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
            elif type(params) is str:
                raise ValueError(("A single parameter was given even " +\
                   "though the distribution given is multivariate " +\
                   "(numparams={}).").format(distribution.numparams)))
            elif type(params) in sequence_types:
                if (len(params) == distribution.numparams):
                    for name in params:
                        self._check_name(name)
                    data_tup = (distribution,\
                        [params[i] for i in range(len(params))], transforms)
                    self._data.append(data_tup)
                else:
                    raise ValueError(("The number of parameters of the " +\
                                     "given distribution ({0:d}) was not " +\
                                     "equal to the number of parameters " +\
                                     "given ({1:d}).").format(\
                                     distribution.numparams, len(params))
            else:
                raise ValueError("The params given to a DistributionSet " +\
                    "(along with a distribution) was not a string nor a " +\
                    "list of strings.")
        else:
            raise ValueError("The distribution given to a DistributionSet " +\
                "was not recognized as a distribution.")
        for iparam in range(distribution.numparams):
            # this line looks weird but it works for any input
            self._params.append(self._data[-1][1][iparam])

    def draw(self, shape=None):
        """
        Draws a point from all distributions.
        
        shape: shape of arrays which are values of return value
        
        returns a dictionary of random values indexed by parameter name
        """
        point = {}
        for idistribution in range(len(self._data)):
            (distribution, params, transforms) = self._data[idistribution]
            if (distribution.numparams == 1):
                point[params[0]] = transforms[0].apply_inverse(\
                    distribution.draw(shape=shape))
            else:
                this_draw = distribution.draw(shape=shape)
                if shape is None:
                    slices = ()
                elif type(shape) in int_types:
                    slices = (slice(None),)
                else:
                    slices = (slice(None),) * len(shape)
                for iparam in range(len(params)):
                    point[params[iparam]] = transforms[iparam].apply_inverse(\
                        this_draw[slices + (iparam,)])
        return point

    def log_value(self, point):
        """
        Evaluates the log of the product of the values of the distributions
        contained in this DistributionSet.
        
        point: should be a dictionary of values indexed by the parameter names
        
        returns: the total log_value coming from contributions from all
                 distributions
        """
        if type(point) is dict:
            result = 0.
            for idistribution in range(len(self._data)):
                (distribution, params, transforms) = self._data[idistribution]
                if (distribution.numparams == 1):
                    result += distribution.log_value(\
                        transforms[0].apply(point[params[0]]))
                else:
                    result += distribution.log_value(\
                        [transforms[i].apply(point[params[i]])\
                                                  for i in range(len(params))])
                for i in range(len(params)):
                    result +=\
                        transforms[i].log_value_addition(point[params[i]])
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
        found = False
        for (this_distribution, these_params, these_transforms) in self._data:
            for iparam in range(len(these_params)):
                if parameter == these_params[iparam]:
                    return\
                        (this_distribution, iparam, these_transforms[iparam])
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
        for idistribution in range(len(self._data)):
            (this_distribution, these_params, these_transforms) =\
                self._data[idistribution]
            if parameter in these_params:
                to_delete = idistribution
                break
        try:
            for par in self._data[to_delete][1]:
                self._params.remove(par)
            self._data = self._data[:to_delete] + self._data[to_delete + 1:]
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
                    if ftfms[iparam] != sparams[iparam]:
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
        if not (type(name) is str):
            raise ValueError("A parameter provided to a DistributionSet " +\
                "was not a string.")
        broken = False
        for idistribution in range(len(self._data)):
            for param in self._data[idistribution]:
                if name == param:
                    broken = True
                    break
            if broken:
                break
        if broken:
            raise ValueError("The name of a parameter provided to a " +\
                "DistributionSet is already taken.")
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data about this DistributionSet.
        Each distribution tuple is saved as a subgroup in the hdf5 file.
        
        group: the hdf5 file group to fill
        """
        for (ituple, (distribution, params, transforms)) in\
            enumerate(self._data):
            subgroup = group.create_group('distribution_{}'.format(ituple))
            distribution.fill_hdf5_group(subgroup)
            for iparam in range(distribution.numparams):
                subgroup.attrs['parameter_{}'.format(iparam)] = params[iparam]
                subsubgroup =\
                    subgroup.create_group('transform_{}'.format(iparam))
                transforms[iparam].fill_hdf5_group(subsubgroup)

