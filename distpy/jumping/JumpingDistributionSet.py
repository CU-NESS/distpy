"""
File: distpy/jumping/JumpingDistributionSet.py
Author: Keith Tauscher
Date: 12 Feb 2018

Description: A container which can hold an arbitrary number of jumping
             distributions, each of which can have any number of parameters
             which it describes (as long as the specific distribution supports
             that number of parameters). Distribution objects can be added
             through
             JumpingDistributionSet.add_distribution(distribution, params)
             where distribution is a JumpingDistribution and params is a list
             of the parameters to which distribution applies. Once all the
             jumping distributions are added, points can be drawn using
             JumpingDistributionSet.draw() and the log_value of the entire set
             of jumping distributions can be evaluated at a point using
             JumpingDistributionSet.log_value(point). See documentation of
             individual functions for further details.
"""
import numpy as np
from ..util import Savable, Loadable, int_types, sequence_types, triangle_plot
from ..transform import cast_to_transform_list, TransformList, TransformSet,\
    NullTransform
from .JumpingDistribution import JumpingDistribution
from .LoadJumpingDistribution import load_jumping_distribution_from_hdf5_group
try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str

class JumpingDistributionSet(Savable, Loadable):
    """
    An object which keeps track of many jumping distributions which can be
    univariate or multivariate. It provides methods like log_value, which calls
    log_value on all of its constituent jumping distributions, and draw, which
    draws from all of its constituent jumping distributions.
    """
    def __init__(self, jumping_distribution_tuples=[]):
        """
        Creates a new JumpingDistributionSet with the given distributions
        inside.
        
        jumping_distribution_tuples: a list of lists/tuples of the form
                                     (distribution, params) where distribution
                                     is an instance of the JumpingDistribution
                                     class and params is a list of parameters
                                     (strings) which distribution describes
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
                        "JumpingDistributionSet was not a sequence of " +\
                        "length 3 like (distribution, params, transforms).")
        else:
            raise ValueError("The jumping_distribution_tuples argument " +\
                "given to the initializer was not list-like. It should be " +\
                "a list of tuples of the form (jumping_distribution, " +\
                "params, transformations) where jumping_distribution is a " +\
                "JumpingDistribution object and params and transformations " +\
                "are lists of strings.")
    
    @property
    def empty(self):
        """
        Finds whether this JumpingDistributionSet is empty.
        
        returns True if no distributions have been added, False otherwise
        """
        return (len(self._data) == 0)
    
    @property
    def params(self):
        """
        Finds and returns the parameters which this JumpingDistributionSet
        describes.
        """
        if not hasattr(self, '_params'):
            self._params = []
        return self._params
    
    @property
    def discrete_params(self):
        """
        Finds and returns the discrete parameters which this
        JumpingDistributionSet describes.
        """
        if not hasattr(self, '_discrete_params'):
            self._discrete_params = []
        return self._discrete_params
    
    @property
    def continuous_params(self):
        """
        Finds and returns the continuous parameters which this
        JumpingDistributionSet describes.
        """
        if not hasattr(self, '_continuous_params'):
            self._continuous_params = []
        return self._continuous_params
    
    @property
    def numparams(self):
        """
        Property storing the number of parameters in this
        JumpingDistributionSet.
        """
        return len(self.params)
    
    def add_distribution(self, distribution, params, transforms=None):
        """
        Adds a jumping_distribution and the parameters it describes to the
        JumpingDistributionSet.
        
        distribution: JumpingDistribution object describing the given
                      parameters
        params list of parameters described by the given distribution
               (can be a single string if the distribution is univariate)
        transforms list of transformations to apply to the parameters
                   (can be a single string if the distribution is univariate)
        """
        if isinstance(distribution, JumpingDistribution):
            transforms = cast_to_transform_list(transforms,\
                num_transforms=distribution.numparams)
            if distribution.numparams == 1:
                if type(params) is str:
                    self._check_name(params)
                    self._data.append((distribution, [params], transforms))
                elif type(params) in sequence_types:
                    if len(params) > 1:
                        raise ValueError("The distribution given to a " +\
                            "JumpingDistributionSet was univariate, but " +\
                            "more than one parameter was given.")
                    else:
                        self._check_name(params[0])
                        self._data.append(\
                            (distribution, [params[0]], transforms))
                else:
                    raise ValueError("The type of the parameters given " +\
                        "to a JumpingDistributionSet was not recognized.")
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
                raise ValueError("The params given to a " +\
                    "JumpingDistributionSet (along with a jumping " +\
                    "distribution) was not a string nor a list of strings.")
        else:
            raise ValueError("The distribution given to a " +\
                "JumpingDistributionSet was not recognized as a jumping " +\
                "distribution.")
        last_distribution_tuple = self._data[-1]
        params_of_last_distribution_tuple = last_distribution_tuple[1]
        self.params.extend(params_of_last_distribution_tuple)
        if distribution.is_discrete:
            self.discrete_params.extend(params_of_last_distribution_tuple)
        else:
            self.continuous_params.extend(params_of_last_distribution_tuple)
    
    def __add__(self, other):
        """
        Adds this JumpingDistributionSet to another.
        
        other: a JumpingDistributionSet object with parameters distinct from
               the parameters of self
        
        returns: JumpingDistributionSet object which is the combination of the
                 given JumpingDistributionSet objects
        """
        if isinstance(other, JumpingDistributionSet):
            if set(self.params) & set(other.params):
                raise ValueError("The two JumpingDistributionSet objects " +\
                    "shared at least one parameter.")
            else:
                return JumpingDistributionSet(\
                    jumping_distribution_tuples=self._data+other._data)
        else:
            raise TypeError("Can only add JumpingDistributionSet objects " +\
                "to other JumpingDistributionSet objects.")
    
    def __iadd__(self, other):
        """
        Adds all distributions from other to this JumpingDistributionSet.
        
        other: JumpingDistributionSet object with parameters distinct from the
               parameters of self
        """
        if isinstance(other, JumpingDistributionSet):
            for distribution_tuple in other._data:
                self.add_distribution(*distribution_tuple)
            return self
        else:
            raise TypeError("JumpingDistributionSet objects can only have " +\
                "other JumpingDistributionSet objects added to them.")
    
    def modify_parameter_names(self, function):
        """
        Modifies the names of the parameters in this distribution by applying
        the given function to each one.
        
        function: function to apply to each parameter name
        """
        self._params = list(map(function, self.params))
        self._discrete_params = list(map(function, self.discrete_params))
        self._continuous_params = list(map(function, self.continuous_params))
        self._data = [(distribution, list(map(function, params)), transforms)\
            for (distribution, params, transforms) in self._data]

    def draw(self, source, shape=None, random=np.random):
        """
        Draws a point from all distributions.
        
        source: dictionary with parameters as keys and source parameters as
                values
        shape: shape of arrays which are values of return value
        
        returns a dictionary of random values indexed by parameter name
        """
        point = {}
        for (distribution, params, transforms) in self._data:
            if (distribution.numparams == 1):
                point[params[0]] = transforms[0].apply_inverse(\
                    distribution.draw(transforms[0].apply(source[params[0]]),\
                    shape=shape, random=random))
            else:
                transformed_source = np.array([transform(source[param])\
                    for (param, transform) in zip(params, transforms)])
                this_draw = distribution.draw(transformed_source, shape=shape,\
                    random=random)
                if type(shape) is type(None):
                    for iparam in range(len(params)):
                        point[params[iparam]] =\
                            transforms[iparam].apply_inverse(this_draw[iparam])
                else:
                    for iparam in range(len(params)):
                        point[params[iparam]] =\
                            transforms[iparam].apply_inverse(\
                            this_draw[...,iparam])
        return point

    def log_value(self, source, destination):
        """
        Evaluates the log of the product of the values of the distributions
        contained in this JumpingDistributionSet from source to destination.
        
        source, destination: should be dictionaries of values indexed by the
                             parameter names
        
        returns: the total log_value coming from contributions from all
                 distributions
        """
        if isinstance(source, dict) and isinstance(destination, dict):
            result = 0.
            for (distribution, params, transforms) in self._data:
                if (distribution.numparams == 1):
                    tsource = transforms[0].apply(source[params[0]])
                    tdest = transforms[0].apply(destination[params[0]])
                    result += distribution.log_value(tsource, tdest)
                else:
                    tsource = []
                    tdest = []
                    for (param, transform) in zip(params, transforms):
                        tsource.append(transform.apply(source[param]))
                        tdest.append(transform.apply(destination[param]))
                    tsource = np.array(tsource)
                    tdest = np.array(tdest)
                    result += distribution.log_value(tsource, tdest)
                for (param, transform) in zip(params, transforms):
                    if isinstance(transform, NullTransform):
                        continue
                    result += transform.log_derivative(destination[param])
            return result
        else:
            raise TypeError("Either source or destinaion given to " +\
                "log_value function of a JumpingDistributionSet was not a " +\
                "dictionary of values indexed by parameter names.")

    def log_value_difference(self, source, destination):
        """
        Evaluates the log of the product of the values of the distributions
        contained in this JumpingDistributionSet from source to destination.
        
        source, destination: should be dictionaries of values indexed by the
                             parameter names
        
        returns: the total log_value coming from contributions from all
                 distributions
        """
        if isinstance(source, dict) and isinstance(destination, dict):
            result = 0.
            for (distribution, params, transforms) in self._data:
                if (distribution.numparams == 1):
                    tsource = transforms[0].apply(source[params[0]])
                    tdest = transforms[0].apply(destination[params[0]])
                    result += distribution.log_value_difference(tsource, tdest)
                else:
                    tsource = []
                    tdest = []
                    for (param, transform) in zip(params, transforms):
                        tsource.append(transform.apply(source[param]))
                        tdest.append(transform.apply(destination[param]))
                    tsource = np.array(tsource)
                    tdest = np.array(tdest)
                    result += distribution.log_value_difference(tsource, tdest)
                for (param, transform) in zip(params, transforms):
                    result += transform.log_derivative(destination[param])
                    result -= transform.log_derivative(source[param])
            return result
        else:
            raise TypeError("Either source or destinaion given to " +\
                "log_value_difference function of a JumpingDistributionSet " +\
                "was not a dictionary of values indexed by parameter names.")

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
            "JumpingDistributionSet was not found.").format(parameter))
    
    def __getitem__(self, parameter):
        """
        Returns the same thing as: self.find_distribution(parameter)
        """
        return self.find_distribution(parameter)
    
    def delete_distribution(self, parameter, throw_error=True):
        """
        Deletes a distribution from this JumpingDistributionSet.
        
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
                    'JumpingDistributionSet.delete_distribution was not in ' +\
                    'the JumpingDistributionSet.')
    
    def __delitem__(self, parameter):
        """
        Deletes the distribution associated with the given parameter. For
        documentation, see delete_distribution function.
        """
        self.delete_distribution(parameter, throw_error=True)
    
    def parameter_strings(self, parameter):
        """
        Makes an informative string about this parameter's place in this
        JumpingDistributionSet.
        
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
        Checks for equality of this JumpingDistributionSet with other. Returns
        True if other has the same distribution_tuples (though they need not
        be internally stored in the same order) and False otherwise.
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
        if isinstance(other, JumpingDistributionSet):
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
        # of the distributions in this JumpingDistributionSet.
        #
        if not isinstance(name, basestring):
            raise ValueError("A parameter provided to a " +\
                "JumpingDistributionSet was not a string.")
        broken = False
        for (distribution, params, transforms) in self._data:
            if name in params:
                broken = True
                break
        if broken:
            raise ValueError("The name of a parameter provided to a " +\
                "JumpingDistributionSet is already taken.")
    
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
        Function which compiles a subset of the JumpingDistribution objects
        in this JumpingDistributionSet: those that represent discrete
        variables.
        
        returns: a JumpingDistributionSet object containing all
                 JumpingDistribution objects in this JumpingDistributionSet
                 which describe discrete variables
        """
        answer = JumpingDistributionSet()
        for (distribution, params, transforms) in self._data:
            if distribution.is_discrete:
                answer.add_distribution(distribution, params, transforms)
        return answer
    
    def continuous_subset(self):
        """
        Function which compiles a subset of the JumpingDistribution objects in
        this JumpingDistributionSet: those that represent continuous variables.
        
        returns: a JumpingDistributionSet object containing all
                 JumpingDistribution objects in this JumpingDistributionSet
                 which describe continuous variables
        """
        answer = JumpingDistributionSet()
        for (distribution, params, transforms) in self._data:
            if not distribution.is_discrete:
                answer.add_distribution(distribution, params, transforms)
        return answer
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data about this
        JumpingDistributionSet. Each distribution tuple is saved as a subgroup
        in the hdf5 file.
        
        group: the hdf5 file group to fill
        """
        for (ituple, distribution_tuple) in enumerate(self._data):
            (distribution, params, transforms) = distribution_tuple
            subgroup = group.create_group('distribution_{}'.format(ituple))
            distribution.fill_hdf5_group(subgroup)
            transforms.fill_hdf5_group(subgroup)
            for iparam in range(distribution.numparams):
                subgroup.attrs['parameter_{}'.format(iparam)] = params[iparam]
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a JumpingDistributionSet from the given hdf5 file group.
        
        group: the group from which to load a JumpingDistributionSet
        
        returns: a JumpingDistributionSet object derived from the given group
        """
        ituple = 0
        jumping_distribution_tuples = []
        while ('distribution_{}'.format(ituple)) in group:
            subgroup = group['distribution_{}'.format(ituple)]
            distribution = load_jumping_distribution_from_hdf5_group(subgroup)
            transform_list = TransformList.load_from_hdf5_group(subgroup)
            params = []
            iparam = 0
            for iparam in range(distribution.numparams):
                params.append(subgroup.attrs['parameter_{}'.format(iparam)])
            jumping_distribution_tuples.append(\
                (distribution, params, transform_list))
            ituple += 1
        return JumpingDistributionSet(\
            jumping_distribution_tuples=jumping_distribution_tuples)
    
    def triangle_plot(self, source, ndraw, parameters=None,\
        in_transformed_space=True, figsize=(8, 8), fig=None, show=False,\
        kwargs_1D={}, kwargs_2D={}, fontsize=28, nbins=100,\
        plot_type='contour', reference_value_mean=None,\
        reference_value_covariance=None, contour_confidence_levels=0.95,\
        parameter_renamer=(lambda x: x)):
        """
        Makes a triangle plot out of ndraw samples from this distribution.
        
        source: dictionary with parameters as keys and source parameters as
                values
        ndraw: integer number of samples to draw to plot in the triangle plot
        parameters: sequence of string parameter names to include in the plot
        figsize: the size of the figure on which to put the triangle plot
        show: if True, matplotlib.pyplot.show is called before this function
                       returns
        kwargs_1D: keyword arguments to pass on to univariate_histogram
                   function
        kwargs_2D: keyword arguments to pass on to bivariate_histogram function
        fontsize: the size of the label fonts
        nbins: the number of bins for each sample
        plot_type: 'contourf', 'contour', or 'histogram'
        reference_value_mean: reference values to place on plots, if there are
                              any
        reference_value_covariance: if not None, used (along with
                                    reference_value_mean) to plot reference
                                    ellipses in each bivariate histogram
        contour_confidence_levels: the confidence level of the contour in the
                                   bivariate histograms. Only used if plot_type
                                   is 'contour' or 'contourf'. Can be single
                                   number or sequence of numbers
        """
        samples = self.draw(source, ndraw)
        if type(parameters) is type(None):
            parameters = self.params
        if in_transformed_space:
            samples = [self.transform_set[parameter](samples[parameter])\
                for parameter in parameters]
        else:
            samples = [samples[parameter] for parameter in parameters]
        labels = [parameter_renamer(parameter) for parameter in parameters]
        return triangle_plot(samples, labels, figsize=figsize, fig=fig,\
            show=show, kwargs_1D=kwargs_1D, kwargs_2D=kwargs_2D,\
            fontsize=fontsize, nbins=nbins, plot_type=plot_type,\
            reference_value_mean=reference_value_mean,\
            reference_value_covariance=reference_value_covariance,\
            contour_confidence_levels=contour_confidence_levels)

