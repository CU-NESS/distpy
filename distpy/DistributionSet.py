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
from .TypeCategories import int_types, sequence_types
from .Saving import Savable
from .Distribution import Distribution

valid_transforms = ['log', 'log10', 'square', 'arcsin', 'logistic']

ln10 = np.log(10)

def _check_if_valid_transform(transform):
    #
    # Checks if the given variable is either None
    # or a string describing a valid transform.
    #
    if type(transform) is str:
        if transform not in valid_transforms:
            raise ValueError("The transform given to apply" +\
                             " to a variable was not valid.")
    elif (transform is not None):
        raise ValueError("The type of the transform given to a " +\
                         "DistributionSet to apply to a parameter was not " +\
                         "recognized.")

def _log_value_addition(value, transform):
    #
    # Finds the term which should be added to the log value of the distribution
    # due to the transform (pretty much the log of the derivative of the
    # transformed parameter with respect to the original parameter evaluated at
    # value.
    #
    if transform is None:
        return 0.
    elif transform == 'log':
        return -1. * np.log(value)
    elif transform == 'log10':
        return -1. * np.log(ln10 * value)
    elif transform == 'square':
        return np.log(2 * value)
    elif transform == 'arcsin':
        return -np.log(1.-np.power(value, 2.)) / 2.
    elif transform == 'logistic':
        return -np.log(value * (1. - value))
    else:
        raise ValueError("For some reason the _log_value_addition " +\
                         "function wasn't implemented for the transform " +\
                         "given, which is \"%s\"." % (transform,))

def _apply_transform(value, transform):
    #
    # Applies the given transform to the value and returns the result.
    #
    if transform is None:
        return value
    elif transform == 'log':
        return np.log(value)
    elif transform == 'log10':
        return np.log10(value)
    elif transform == 'square':
        return np.power(value, 2.)
    elif transform == 'arcsin':
        return np.arcsin(value)
    elif transform == 'logistic':
        return np.log(value / (1. - value))
    else:
        raise ValueError("Something went wrong and an attempt to evaluate " +\
                         "an invalid transform was made. This should " +\
                         "have been caught by previous error catching!")

def _apply_inverse_transform(value, transform):
    #
    # Applies the inverse of the given transform
    # to the value and returns the result.
    #
    if transform is None:
        return value
    elif transform == 'log':
        return np.exp(value)
    elif transform == 'log10':
        return np.power(10, value)
    elif transform == 'square':
        return np.sqrt(value)
    elif transform == 'arcsin':
        return np.sin(value)
    elif transform == 'logistic':
        return 1 / (1. + (np.exp(-value)))
    else:
        raise ValueError("Something went wrong and an attempt to evaluate" +\
                         " an invalid (inverse) transform was made. This" +\
                         "should've been caught by previous error catching!")
        

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
            for idistribution in xrange(len(distribution_tuples)):
                this_tup = distribution_tuples[idistribution]
                if (type(this_tup) in sequence_types) and len(this_tup) == 3:
                    (distribution, params, transforms) =\
                        distribution_tuples[idistribution]
                    self.add_distribution(distribution, params, transforms)
                else:
                    raise ValueError("One of the distribution tuples " +\
                                     "provided to the initializer of a " +\
                                     "DistributionSet was not a sequence " +\
                                     "of length 3 like " +\
                                     "(distribution, params, transforms).")
        else:
            raise ValueError("The distribution_tuples argument given to " +\
                             "the initializer was not list-like. It should " +\
                             "be a list of tuples of the form " +\
                             "(distribution, params, transformations) " +\
                             "where distribution is a Distribution " +\
                             "object and params and transformations " +\
                             "are lists of strings.")

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
                transforms = [None] * distribution.numparams
            elif (type(transforms) is str):
                _check_if_valid_transform(transforms)
                if (distribution.numparams == 1):
                    transforms = [transforms]
                else:
                    raise ValueError("The transforms variable applied" +\
                                     " to parameters of a DistributionSet " +\
                                     "was provided as a string even though " +\
                                     "the distribution being provided was " +\
                                     "multivariate.")
            elif type(transforms) in sequence_types:
                if len(transforms) == distribution.numparams:
                    for itransform in range(len(transforms)):
                        _check_if_valid_transform(transforms[itransform])
                else:
                    raise ValueError("The list of transforms applied to " +\
                                     "parameters in a DistributionSet was " +\
                                     "not the same length as the list of " +\
                                     "parameters of the distribution.")
            else:
                raise ValueError("The type of the transforms variable " +\
                                 "supplied to DistributionSet's " +\
                                 "add_distribution function was not " +\
                                 "recognized. It should be a single valid " +\
                                 "string (if distribution is univariate) " +\
                                 "or list of valid strings (if " +\
                                 "distribution is multivariate).")
            if distribution.numparams == 1:
                if type(params) is str:
                    self._check_name(params)
                    self._data.append((distribution, [params], transforms))
                elif type(params) in sequence_types:
                    if len(params) > 1:
                        raise ValueError("The distribution given to a " +\
                                         "DistributionSet was univariate, " +\
                                         "but more than one parameter was " +\
                                         "given.")
                    else:
                        self._check_name(params[0])
                        self._data.append(\
                            (distribution, [params[0]], transforms))
                else:
                    raise ValueError("The type of the parameters given " +\
                                     "to a DistributionSet was not " +\
                                     "recognized.")
            elif type(params) is str:
                raise ValueError("A single parameter was given even though" +\
                                 " the distribution given is multivariate (" +\
                                 "numparams=" +\
                                 ("%i)." % (distribution.numparams,)))
            elif type(params) in sequence_types:
                if (len(params) == distribution.numparams):
                    for name in params:
                        self._check_name(name)
                    data_tup = (distribution,\
                        [params[i] for i in range(len(params))], transforms)
                    self._data.append(data_tup)
                else:
                    raise ValueError("The number of parameters of the " +\
                                     "given distribution (" +\
                                     ("%i) " % (distribution.numparams,)) +\
                                     "was not equal to the number of " +\
                                     "parameters given (" +\
                                     ("%i)." % (len(params),)))
            else:
                raise ValueError("The params given to a DistributionSet" +\
                                 " (along with a distribution) was not " +\
                                 "a string nor a list of strings.")
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
                point[params[0]] = _apply_inverse_transform(\
                    distribution.draw(shape=shape), transforms[0])
            else:
                this_draw = distribution.draw(shape=shape)
                if shape is None:
                    slices = ()
                elif type(shape) in int_types:
                    slices = (slice(None),)
                else:
                    slices = (slice(None),) * len(shape)
                for iparam in xrange(len(params)):
                    point[params[iparam]] = _apply_inverse_transform(\
                        this_draw[slices + (iparam,)], transforms[iparam])
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
                        _apply_transform(point[params[0]], transforms[0]))
                else:
                    result += distribution.log_value(\
                        [_apply_transform(point[params[i]], transforms[i])\
                                                  for i in range(len(params))])
                for i in range(len(params)):
                    result +=\
                        _log_value_addition(point[params[i]], transforms[i])
            return result
        else:
            raise ValueError("point given to log_value function of a " +\
                             "DistributionSet was not a dictionary of " +\
                             "values indexed by parameter names.")

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
        raise ValueError(("The parameter searched for (%s) " % (parameter,)) +\
                         "in a DistributionSet was not found.")
    
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
                                 'DistributionSet.delete_distribution was ' +\
                                 'not in the DistributionSet.')
    
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
        
        returns (param_string, transform) in tuple form
        """
        string = ""
        (distribution, index, transform) = self.find_distribution(parameter)
        if distribution.numparams != 1:
            string += (self._numerical_adjective(index) + ' param of ')
        string += distribution.to_string()
        return (string, transform)
    
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
        if (type(num) in [int, np.int8, np.int16, np.int32, np.int64]) and\
            (num >= 0):
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
            raise ValueError("Numerical adjectives apply " +\
                             "only to non-negative integers.")

    def _check_name(self, name):
        #
        # Checks the given name to see if it is already taken in the parameters
        # of the distributions in this DistributionSet.
        #
        if not (type(name) is str):
            raise ValueError("A parameter provided to a " +\
                             "DistributionSet was not a string.")
        broken = False
        for idistribution in xrange(len(self._data)):
            for param in self._data[idistribution]:
                if name == param:
                    broken = True
                    break
            if broken:
                break
        if broken:
            raise ValueError("The name of a parameter provided" +\
                             " to a DistributionSet is already taken.")
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data about this DistributionSet.
        Each distribution tuple is saved as a subgroup in the hdf5 file.
        
        group: the hdf5 file group to fill
        """
        for (ituple, (distribution, params, transforms)) in\
            enumerate(self._data):
            subgroup = group.create_group('distribution_%i' % (ituple,))
            distribution.fill_hdf5_group(subgroup)
            for iparam in xrange(distribution.numparams):
                subgroup.attrs['parameter_%i' % (iparam,)] = params[iparam]
                if transforms[iparam] is not None:
                    subgroup.attrs['transformation_%i' % (iparam)] =\
                        transforms[iparam]

