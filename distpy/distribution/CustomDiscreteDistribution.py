"""
File: distpy/distribution/CustomDiscreteDistribution.py
Author: Keith Tauscher
Date: 24 Feb 2018

Description: File containing a class representing a custom n-dimensional
             discrete distribution with finite support.
"""
import numpy as np
from ..util import sequence_types, create_hdf5_dataset, get_hdf5_value
from .Distribution import Distribution

class CustomDiscreteDistribution(Distribution):
    """
    Class representing a custom n-dimensional discrete distribution with finite
    support.
    """
    def __init__(self, variable_values, probability_mass_function,\
        metadata=None):
        """
        variable_values: must be a sequence of 1D numpy.ndarrays representing
                         the possible values of each given parameter
        probability_mass_function: the probabilities of each combination of
                                   possible values in a ND numpy.ndarray with
                                   shape given by the addition of the shapes of
                                   the arrays in variable_values
        metadata: data to store alongside this distribution
        """
        self.variable_values = variable_values
        self.probability_mass_function = probability_mass_function
        self.metadata = metadata
    
    @property
    def variable_values(self):
        """
        Property storing a sequence of 1D numpy.ndarrays each containing the
        possible values of each of this distribution's variables.
        """
        if not hasattr(self, '_variable_values'):
            raise AttributeError("variable_values referenced before it was " +\
                "set.")
        return self._variable_values
    
    @variable_values.setter
    def variable_values(self, value):
        """
        Sets the variable values allowed by this distribution.
        
        value: if this distribution is 1D, then value can be either a single 1D
               numpy.ndarray or a list containing a single 1D numpy.ndarray. if
               this distribution is nD, then value must be a length-n sequence
               of 1D numpy.ndarrays giving the possible values of the ith
               variable described by this distribution
        """
        if isinstance(value, np.ndarray) and (value.ndim == 1):
            value = [value]
        if type(value) in sequence_types:
            if all([isinstance(element, np.ndarray) for element in value]):
                if all([(element.ndim == 1) for element in value]):
                    self._variable_values = [element for element in value]
                else:
                    raise TypeError("Not all numpy.ndarrays in " +\
                        "variable_values were 1 dimensional.")
            else:
                raise TypeError("Not all elements of variable_values were "+\
                    "numpy.ndarrays.")
        else:
            raise TypeError("variable_values was set to neither a 1D " +\
                "numpy.ndarray or a sequence of 1D numpy.ndarrays")
    
    @property
    def shape(self):
        """
        Property storing the shape of the grid used for this
        CustomDiscreteDistribution. It should be the shape of the
        probability_mass_function array.
        """
        if not hasattr(self, '_shape'):
            self._shape =\
                sum([(len(axis),) for axis in self.variable_values], ())
        return self._shape
    
    @property
    def probability_mass_function(self):
        """
        Property storing a n-dimensional numpy.ndarray of shape given by the
        sum of the shapes of the variable_values storing the probabilities of
        each possible combination of variables.
        """
        if not hasattr(self, '_probability_mass_function'):
            raise AttributeError("probability_mass_function was referenced " +\
                "before it was set.")
        return self._probability_mass_function
    
    @probability_mass_function.setter
    def probability_mass_function(self, value):
        """
        Setter for the probability mass function defining this distribution.
        
        value: a numpy.ndarray of shape given by sum of shapes of
               variable_values
        """
        if isinstance(value, np.ndarray):
            if value.shape == self.shape:
                unnormalized_cmf = np.cumsum(value.flatten())
                norm_factor = 1. / unnormalized_cmf[-1]
                self._probability_mass_function = value * norm_factor
                self._flattened_cumulative_mass_function =\
                    unnormalized_cmf * norm_factor
            else:
                raise ValueError(("The shape of the given " +\
                    "probability_mass_function ({0!s}) was not the " +\
                    "expected shape ({1!s}).").format(value.shape, self.shape))
        else:
            raise TypeError("probability_mass_function was not a " +\
                "numpy.ndarray.")
    
    @property
    def flattened_cumulative_mass_function(self):
        """
        Property storing the flattened, normalized, cumulative mass function
        corresponding this distribution's probability mass function. This
        property is set in the setter for the probability_mass_function
        property. This array is importantly used in the process of drawing
        random values from this distribution.
        """
        if not hasattr(self, '_flattened_cumulative_mass_function'):
            raise AttributeError("flattened_cumulative_mass_function " +\
                "referenced before it was set.")
        return self._flattened_cumulative_mass_function
    
    def draw(self, shape=None, random=np.random):
        """
        Draws a point from the distribution. Must be implemented by any base
        class.
        
        shape: if None, returns single random variate
                        (scalar for univariate ; 1D array for multivariate)
               if int, n, returns n random variates
                          (1D array for univariate ; 2D array for multivariate)
               if tuple of n ints, returns that many random variates
                                   n-D array for univariate ;
                                   (n+1)-D array for multivariate
        random: the random number generator to use (default: numpy.random)
        
        returns: either single value (if distribution is 1D) or array of values
        """
        none_shape = (shape is None)
        if none_shape:
            shape = 1
        if isinstance(shape, int):
            shape = (shape,)
        random_values = random.rand(*shape)
        flattened_locations = np.searchsorted(\
            self.flattened_cumulative_mass_function, random_values)
        unpacked_locations = np.unravel_index(flattened_locations, self.shape)
        draws = np.ndarray(shape + (self.numparams,))
        for (iparam, locations) in enumerate(unpacked_locations):
            draws[...,iparam] = self.variable_values[iparam][locations]
        if none_shape:
            draws = draws[0]
        if self.numparams == 1:
            return draws[...,0]
        else:
            return draws
    
    def log_value(self, point):
        """
        Computes the logarithm of the value of this distribution at the given
        point.
        
        point: either single value (if distribution is 1D) or array of values
        
        returns: single number, logarithm of value of this distribution at the
                 given point
        """
        if self.numparams == 1:
            point = [point]
        multi_index = ()
        for (component, axis) in zip(point, self.variable_values):
            try:
                multi_index =\
                    multi_index + (np.where(component == axis)[0][0],)
            except:
                return -np.inf
        return np.log(self.probability_mass_function[multi_index])
    
    @property
    def gradient_computable(self):
        """
        Property which stores whether the gradient of the given distribution
        has been implemented.
        """
        return False
    
    @property
    def hessian_computable(self):
        """
        Property which stores whether the hessian of the given distribution
        has been implemented.
        """
        return False
    
    @property
    def numparams(self):
        """
        Property storing the integer number of parameters described by this
        distribution.
        """
        if not hasattr(self, '_numparams'):
            self._numparams = len(self.variable_values)
        return self._numparams
    
    def to_string(self):
        """
        Returns a string representation of this distribution.
        """
        return "{}-dim custom discrete".format(self.numparams)
    
    def __eq__(self, other):
        """
        Tests for equality between this CustomDiscreteDistribution and other.
        
        other: Distribution with which to check for equality
        
        returns: True or False
        """
        if isinstance(other, CustomDiscreteDistribution):
            if self.numparams == other.numparams:
                for iparam in range(self.numparams):
                    saxis = self.variable_values[iparam]
                    oaxis = other.variable_values[iparam]
                    if np.any(saxis != oaxis):
                        return False
                return np.allclose(self.probability_mass_function,\
                    other.probability_mass_function, rtol=0, atol=1e-10)
            else:
                return False
        else:
            return False
    
    @property
    def is_discrete(self):
        """
        Property storing a boolean describing whether this distribution is
        discrete (True) or continuous (False).
        """
        return True
    
    def fill_hdf5_group(self, group, save_metadata=True, pmf_link=None,\
        **axis_links):
        """
        Fills the given hdf5 file group with information about this
        CustomDiscreteDistribution.
        
        group: hdf5 file group to fill with information about this distribution
        save_metadata: if True, attempts to save metadata alongside
                                distribution and throws error if it fails
                       if False, metadata is ignored in saving process
        pmf_link: if None, probability_mass_function is saved directly in the
                           probability_mass_function group
                  otherwise, pmf_link should be a link to a extant location
                             where the probability_mass_function is already
                             stored
        axis_links: dictionary (default empty) whose keys are strings of the
                    form 'variable{0:d}'.format(variable_number) and whose
                    values are either None or links to existing locations where
                    the axis values are already saved
        """
        group.attrs['class'] = 'CustomDiscreteDistribution'
        create_hdf5_dataset(group, 'probability_mass_function',\
            data=self.probability_mass_function, link=pmf_link)
        subgroup = group.create_group('variable_values')
        for (iaxis, axis) in enumerate(self.variable_values):
            string_name = 'variable{0:d}'.format(iaxis)
            if string_name in axis_links:
                link = axis_links[string_name]
            else:
                link = None
            create_hdf5_dataset(subgroup, string_name, data=axis, link=link)
        if save_metadata:
            self.save_metadata(group)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a CustomDiscreteDistribution from the given hdf5 file group.
        
        group: the same hdf5 file group which fill_hdf5_group was called on
               when this CustomDiscreteDistribution was saved
        
        returns: a CustomDiscrete Distribution object created from the
                 information in the given group
        """
        try:
            assert group.attrs['class'] == 'CustomDiscreteDistribution'
        except:
            raise TypeError("The given hdf5 file doesn't seem to contain a " +\
                "CustomDiscreteDistribution.")
        metadata = Distribution.load_metadata(group)
        variable_values = []
        ivar = 0
        subgroup = group['variable_values']
        while 'variable{0:d}'.format(ivar) in subgroup:
            axis = get_hdf5_value(subgroup['variable{0:d}'.format(ivar)])
            variable_values.append(axis)
            ivar += 1
        probability_mass_function =\
            get_hdf5_value(group['probability_mass_function'])
        return CustomDiscreteDistribution(variable_values,\
            probability_mass_function)
    
    @property
    def can_give_confidence_intervals(self):
        """
        This distribution cannot yield confidence intervals.
        """
        return False
    
    def copy(self):
        """
        Returns a deep copy of this Distribution. This function ignores
        metadata.
        """
        return CustomDiscreteDistribution(self.variable_values.copy(),\
            self.probability_mass_function.copy())

