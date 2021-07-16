"""
Module containing a container which can hold an arbitrary number of
`distpy.jumping.JumpingDistribution.JumpingDistribution` objects, each of which
can have any number of parameters which it describes (as long as the specific
`distpy.jumping.JumpingDistribution.JumpingDistribution` supports that number
of parameters). `distpy.jumping.JumpingDistribution.JumpingDistribution`
objects can be added through `JumpingDistributionSet.add_distribution`. Once
all the distributions are added, points can be drawn using the
`JumpingDistributionSet.draw` method and the log value of the entire set of
distributions can be evaluated at a given source and destination using the
`JumpingDistributionSet.log_value` and
`JumpingDistributionSet.log_value_difference` methods. See documentation of
individual methods for further details. This class represents a dictionary- or
set-like container of `distpy.jumping.JumpingDistribution.JumpingDistribution`
objects; see `distpy.jumping.JumpingDistributionList.JumpingDistributionList`
for a list-like container of
`distpy.jumping.JumpingDistribution.JumpingDistribution` objects.

**File**: $DISTPY/distpy/jumping/JumpingDistributionSet.py  
**Author**: Keith Tauscher  
**Date**: 3 Jul 2021
"""
import numpy as np
from ..util import Savable, Loadable, int_types, sequence_types, triangle_plot
from ..transform import NullTransform, TransformList, TransformSet
from .JumpingDistribution import JumpingDistribution
from .JumpingDistributionList import JumpingDistributionList
from .LoadJumpingDistribution import load_jumping_distribution_from_hdf5_group
try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str

class JumpingDistributionSet(Savable, Loadable):
    """
    A container which can hold an arbitrary number of
    `distpy.jumping.JumpingDistribution.JumpingDistribution` objects, each of
    which can have any number of parameters which it describes (as long as the
    specific `distpy.jumping.JumpingDistribution.JumpingDistribution` supports
    that number of parameters).
    `distpy.jumping.JumpingDistribution.JumpingDistribution` objects can be
    added through `JumpingDistributionSet.add_distribution`. Once all the
    distributions are added, points can be drawn using the
    `JumpingDistributionSet.draw` method and the log value of the entire set of
    distributions can be evaluated at a given source and destination using the
    `JumpingDistributionSet.log_value` and
    `JumpingDistributionSet.log_value_difference` methods. See documentation of
    individual methods for further details. This class represents a dictionary-
    or set-like container of
    `distpy.jumping.JumpingDistribution.JumpingDistribution` objects; see
    `distpy.jumping.JumpingDistributionList.JumpingDistributionList` for a
    list-like container of
    `distpy.jumping.JumpingDistribution.JumpingDistribution` objects.
    """
    def __init__(self, jumping_distribution_tuples=[]):
        """
        Creates a new `JumpingDistributionSet` with the given distributions
        inside.
        
        Parameters
        ----------
        jumping_distribution_tuples : sequence
            a list of sequences of the form
            `(distribution, params, transforms)` or `(distribution, params)`
            where:
            
            - `distribution` is a
            `distpy.jumping.JumpingDistribution.JumpingDistribution`object
            - `params` is a list of string parameter names `distribution`
            describes (or a single string if `distribution` is univariate)
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
        Boolean describing whether this `JumpingDistributionSet` is empty.
        """
        return (len(self._data) == 0)
    
    @property
    def params(self):
        """
        List of string names of parameters which this `JumpingDistributionSet`
        describes.
        """
        if not hasattr(self, '_params'):
            self._params = []
        return self._params
    
    @property
    def discrete_params(self):
        """
        List of string parameter names which are described by the discrete part
        of this `JumpingDistributionSet`.
        """
        if not hasattr(self, '_discrete_params'):
            self._discrete_params = []
        return self._discrete_params
    
    @property
    def continuous_params(self):
        """
        List of string parameter names which are described by the continuous
        part of this `JumpingDistributionSet`.
        """
        if not hasattr(self, '_continuous_params'):
            self._continuous_params = []
        return self._continuous_params
    
    @property
    def numparams(self):
        """
        The integer number of parameters described by this
        `JumpingDistributionSet`.
        """
        return len(self.params)
    
    def __len__(self):
        """
        Function allowing users to access the number of parameters described by
        this `JumpingDistributionSet` by using the built-in `len` function and
        not explicitly referencing `JumpingDistributionSet.numparams`.
        """
        return self.numparams
    
    def add_distribution(self, distribution, params, transforms=None):
        """
        Adds a `distpy.jumping.JumpingDistribution.JumpingDistribution` and the
        parameters it describes to the `JumpingDistributionSet`.
        
        Parameters
        ----------
        distribution : `distpy.jumping.JumpingDistribution.JumpingDistribution`
            distribution describing how the given parameters jump
        params : sequence
            sequence of string names of the parameters described by
            `distribution` (can be a single string if the `distribution` is
            univariate)
        transforms : `distpy.transform.TransformList.TransformList` or\
        `distpy.transform.Transform.Transform` or sequence or None
            list of transformations to apply to the parameters (can be a single
            string if `distribution` is univariate)
        """
        if isinstance(distribution, JumpingDistribution):
            transforms = TransformList.cast(transforms,\
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
        Adds this `JumpingDistributionSet` to another by multiplying their PDFs
        and assuming the parameters of `self` and `other` are independent.
        Allows for use of the `+` operator on `JumpingDistributionSet` objects.
        
        Parameters
        ----------
        other : `JumpingDistributionSet`
            a `JumpingDistributionSet` with parameters distinct from the
            parameters of `self`
        
        Returns
        -------
        combined : `JumpingDistributionSet`
            combination of the given `JumpingDistributionSet` objects
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
        Adds all distributions from `other` to this `JumpingDistributionSet`.
        Allows for use of the `+=` operator with `JumpingDistributionSet`
        objects.
        
        Parameters
        ----------
        other : `JumpingDistributionSet`
            `JumpingDistributionSet` with parameters distinct from the
            parameters of `self`
        
        Returns
        -------
        self : `JumpingDistributionSet`
            this object with its state changed to include distributions from
            `other`
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
        
        Parameters
        ----------
        function : Callable
            function to apply to each parameter name
        """
        self._params = list(map(function, self.params))
        self._discrete_params = list(map(function, self.discrete_params))
        self._continuous_params = list(map(function, self.continuous_params))
        self._data = [(distribution, list(map(function, params)), transforms)\
            for (distribution, params, transforms) in self._data]

    def draw(self, source, shape=None, random=np.random):
        """
        Draws a destination from all distributions given the source.
        
        Parameters
        ----------
        source : dict
            dictionary with parameters as keys and source parameters as values
        shape : tuple or int or None
            - if `shape` is None, the values of the returned destination
            dictionary are numbers
            - if `shape` is an int, the values of the returned destination
            dictionary are 1D arrays of length `shape`
            - if `shape` is a tuple, the values of the returned destination
            dictionary are arrays of shape `shape`
        
        Returns
        -------
        destination : dict
            dictionary of random values indexed by parameter name. The shape of
            the values of the dictionary are determined by `shape`
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
        contained in this `JumpingDistributionSet` from `source` to
        `destination`.
        
        Parameters
        ----------
        source : dict
            dictionary with parameters as keys and source parameters as values
        destination : dict
            dictionary with parameters as keys and destination parameters as
            values
        
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
        Evaluates the log of the product of the ratios of the distributions
        contained in this `JumpingDistributionSet` from source to destination.
        
        Parameters
        ----------
        source : dict
            dictionary with parameters as keys and source parameters as values
        destination : dict
            dictionary with parameters as keys and destination parameters as
            values
        
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
        applied to the parameter. Throws a `ValueError` if `parameter` is not
        described by this `JumpingDistributionSet`.
        
        Parameters
        ----------
        parameter : str
            string name of parameter to search for
        
        Returns
        -------
        result : tuple
            a tuple of the form `(distribution, index, transform)`, where:
            
            - `distribution` is the
            `distpy.jumping.JumpingDistribution.JumpingDistribution` that
            applies to the given parameter
            - `index` is the parameter index of the `parameter` in
            `distribution`
            - `transform` is the `distpy.transform.Transform.Transform` object
            that applies to the given parameter
        """
        for (jumping_distribution, params, transforms) in self._data:
            for (iparam, param) in enumerate(params):
                if parameter == param:
                    return (jumping_distribution, iparam, transforms[iparam])
        raise ValueError(("The parameter searched for ({!s}) in a " +\
            "JumpingDistributionSet was not found.").format(parameter))
    
    def __getitem__(self, parameter):
        """
        Finds the distribution associated with the given parameter. Also finds
        the index of the parameter in that distribution and the transformation
        applied to the parameter. Throws a `ValueError` if `parameter` is not
        described by this `JumpingDistributionSet`. Alias for
        `JumpingDistributionSet.find_distribution` that allows for square
        bracket indexing of `JumpingDistributionSet` objects
        
        Parameters
        ----------
        parameter : str
            string name of parameter to search for
        
        Returns
        -------
        result : tuple
            a tuple of the form `(distribution, index, transform)`, where:
            
            - `distribution` is the
            `distpy.jumping.JumpingDistribution.JumpingDistribution` that
            applies to the given parameter
            - `index` is the parameter index of the `parameter` in
            `distribution`
            - `transform` is the `distpy.transform.Transform.Transform` object
            that applies to the given parameter
        """
        return self.find_distribution(parameter)
    
    def delete_distribution(self, parameter, throw_error=True):
        """
        Deletes a distribution from this `JumpingDistributionSet`.
        
        Parameters
        ----------
        parameter : str
            any parameter in the distribution to delete
        throw_error : bool
            - if `throw_error` is True (default), an error is thrown if
            `parameter` is not found
            - if `throw_error` is False, this method does nothing if
            `parameter` is not found
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
        Deletes a distribution from this `JumpingDistributionSet`. Alias for
        `JumpingDistributionSet.delete_distribution` with `throw_error=True`
        that allows for the use of the `del` keyword with
        `JumpingDistributionSet` objects
        
        Parameters
        ----------
        parameter : str
            any parameter in the distribution to delete
        """
        self.delete_distribution(parameter, throw_error=True)
    
    def __eq__(self, other):
        """
        Checks for equality of this `JumpingDistributionSet` with `other`.
        
        Parameters
        ----------
        other : object
            object with which to check for equality
        
        Returns
        -------
        result : bool
            True if and only if other is another `JumpingDistributionSet` that
            has the same distribution tuples (though they need not be
            internally stored in the same order).
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
        Checks for inequality of this `JumpingDistributionSet` with `other`.
        
        Parameters
        ----------
        other : object
            object with which to check for inequality
        
        Returns
        -------
        result : bool
            False if and only if other is another `JumpingDistributionSet` that
            has the same distribution tuples (though they need not be
            internally stored in the same order).
        """
        return (not self.__eq__(other))
    
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
        The `distpy.transform.TransformSet.TransformSet` object describing the
        transformations of the parameters in this `JumpingDistributionSet`.
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
        Function which compiles a subset of the
        `distpy.jumping.JumpingDistribution.JumpingDistribution` objects in
        this `JumpingDistributionSet`: those that represent discrete variables.
        
        Returns
        -------
        subset : `JumpingDistributionSet`
            distribution containing all
            `distpy.jumping.JumpingDistribution.JumpingDistribution` objects in
            this `JumpingDistributionSet` which describe discrete variables
        """
        answer = JumpingDistributionSet()
        for (distribution, params, transforms) in self._data:
            if distribution.is_discrete:
                answer.add_distribution(distribution, params, transforms)
        return answer
    
    def continuous_subset(self):
        """
        Function which compiles a subset of the
        `distpy.jumping.JumpingDistribution.JumpingDistribution` objects in
        this `JumpingDistributionSet`: those that represent continuous
        variables.
        
        Returns
        -------
        subset : `JumpingDistributionSet`
            distribution containing all
            `distpy.jumping.JumpingDistribution.JumpingDistribution` objects in
            this `JumpingDistributionSet` which describe continuous variables
        """
        answer = JumpingDistributionSet()
        for (distribution, params, transforms) in self._data:
            if not distribution.is_discrete:
                answer.add_distribution(distribution, params, transforms)
        return answer
    
    def jumping_distribution_list(self, parameters):
        """
        Creates a
        `distpy.jumping.JumpingDistributionList.JumpingDistributionList` out of
        this `JumpingDistributionSet` by ordering it in the same way as the
        given parameters.
        
        Parameters
        ----------
        parameters : sequence
            the string names of the parameters whose distribution should be put
            into the list, including order. May oy may not contain all of this
            `JumpingDistributionSet` object's parameters
        
        Returns
        -------
        list_form : `distpy.jumping.JumpingDistributionList.JumpingDistributionList`
            object containing the distribution of the given parameters
        """
        to_list = [parameter for parameter in parameters]
        distribution_order = []
        while to_list:
            first_parameter = to_list[0]
            broken = False
            for (ituple, (jumping_distribution, params, transforms)) in\
                enumerate(self._data):
                if (first_parameter in params):
                    if ituple in distribution_order:
                        raise ValueError("The same jumping distribution " +\
                            "cannot be put in the same " +\
                            "JumpingDistributionList twice using this " +\
                            "function. The parameters must be out of order.")
                    distribution_order.append(ituple)
                    broken = True
                    break
            if not broken:
                raise ValueError(("The parameter {!s} was not found in " +\
                    "this JumpingDistributionSet, so couldn't be used to " +\
                    "populate a JumpingDistributionList object.").format(\
                    first_parameter))
            (jumping_distribution, params, transforms) =\
                self._data[distribution_order[-1]]
            for (distribution_param, list_param) in\
                zip(params, to_list[:jumping_distribution.numparams]):
                if distribution_param != list_param:
                    raise ValueError("Something went wrong. You must have " +\
                        "parameters out of order with respect to the " +\
                        "distributions they are in.")
            to_list = to_list[jumping_distribution.numparams:]
        jumping_distribution_tuples =\
            [self._data[element][::2] for element in distribution_order]
        return JumpingDistributionList(\
            jumping_distribution_tuples=jumping_distribution_tuples)
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data about this
        `JumpingDistributionSet`. Each distribution tuple is saved as a
        subgroup in the hdf5 file.
        
        Parameters
        ----------
        group : h5py.Group
            the hdf5 file group to fill
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
        Loads a `JumpingDistributionSet` from the given hdf5 file group.
        
        Parameters
        ----------
        group : h5py.Group
            the group from which to load a `JumpingDistributionSet`
        
        Returns
        -------
        returns : `JumpingDistributionSet`
            a `JumpingDistributionSet` object loaded from the given group
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
        plot_type='contour', plot_limits=None, reference_value_mean=None,\
        reference_value_covariance=None, contour_confidence_levels=0.95,\
        parameter_renamer=(lambda x: x), tick_label_format_string='{x:.3g}',\
        num_ticks=3, minor_ticks_per_major_tick=1, xlabel_rotation=0,\
        xlabelpad=None, ylabel_rotation=90, ylabelpad=None):
        """
        Makes a triangle plot out of ndraw samples from this distribution.
        
        Parameters
        ----------
        source : dict
            dictionary with parameters as keys and source parameters as values
        ndraw : int
            the integer number of samples to draw to plot in the triangle plot
        parameters : sequence
            sequence of string parameter names to include in the plot
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
        samples = self.draw(source, ndraw)
        if type(parameters) is type(None):
            parameters = self.params
        if in_transformed_space:
            samples = [self.transform_set[parameter](samples[parameter])\
                for parameter in parameters]
        else:
            samples = [samples[parameter] for parameter in parameters]
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
            tick_label_format_string=tick_label_format_string,\
            xlabel_rotation=xlabel_rotation, xlabelpad=xlabelpad,\
            ylabel_rotation=ylabel_rotation, ylabelpad=ylabelpad)

