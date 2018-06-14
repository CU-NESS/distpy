"""
File: distpy/jumping/JumpingDistributionSum.py
Author: Keith Tauscher
Date: 13 Jun 2018

Description: File containing class which represents a weighted sum of
             distributions.
"""
import numpy as np
from ..util import int_types, sequence_types
from .JumpingDistribution import JumpingDistribution

rand = np.random

class JumpingDistributionSum(JumpingDistribution):
    """
    Class which represents a weighted sum of JumpingDistribution objects.
    """
    def __init__(self, jumping_distributions, weights):
        """
        Creates a new JumpingDistributionSum object out of the given
        JumpingDistribution objects.
        
        jumping_distributions: sequence of JumpingDistribution objects with the
                               same numparams
        weights: sequence of numbers with which to combine (need not be
                 normalized but they must all be positive)
        """
        self.jumping_distributions = jumping_distributions
        self.weights = weights
    
    @property
    def jumping_distributions(self):
        """
        Property storing a list of JumpingDistribution objects which make up
        this JumpingDistributionSum.
        """
        if not hasattr(self, '_jumping_distributions'):
            raise AttributeError("jumping_distributions was referenced " +\
                "before it was set.")
        return self._jumping_distributions
    
    @jumping_distributions.setter
    def jumping_distributions(self, value):
        """
        Setter for the JumpingDistribution objects making up this
        JumpingDistributionSum.
        
        value: sequence of JumpingDistribution objects making up this
               JumpingDistributionSum
        """
        if type(value) in sequence_types:
            if all([isinstance(element, JumpingDistribution)\
                for element in value]):
                unique_nums_of_parameters =\
                    np.unique([element.numparams for element in value])
                unique_is_discretes =\
                    np.unique([element.is_discrete for element in value])
                if (len(unique_nums_of_parameters) == 1) and\
                    (len(unique_is_discretes) == 1):
                    self._jumping_distributions =\
                        [element for element in value]
                    self._numparams = unique_nums_of_parameters[0]
                    self._is_discrete = unique_is_discretes[0]
                else:
                    raise ValueError("Not all jumping_distributions added " +\
                        "together here had the same number of parameters.")
            else:
                raise TypeError("At least one element of the " +\
                    "jumping_distributions sequence was not a " +\
                    "JumpingDistribution object.")
        else:
            raise TypeError("jumping_distributions was set to a non-sequence.")
    
    @property
    def num_jumping_distributions(self):
        """
        Property storing the number of JumpingDistribution objects which make
        up this JumpingDistributionSum object.
        """
        if not hasattr(self, '_num_jumping_distributions'):
            self._num_jumping_distributions = len(self.jumping_distributions)
        return self._num_jumping_distributions
    
    @property
    def weights(self):
        """
        Property storing the weights with which to combine the distributions.
        """
        if not hasattr(self, '_weights'):
            raise AttributeError("weights was referenced before it was set.")
        return self._weights
    
    @weights.setter
    def weights(self, value):
        """
        Setter for the weights with which to combine the distributions.
        
        value: sequence of numbers
        """
        if type(value) in sequence_types:
            if len(value) == self.num_jumping_distributions:
                value = np.array(value)
                if np.all(np.zeros(self.num_jumping_distributions) < value):
                    self._weights = value
                else:
                    raise ValueError("At least one weight was non-positive.")
            else:
                raise ValueError("The length of the weights sequence was " +\
                    "not the same as the length of the " +\
                    "jumping_distributions sequence.")
        else:
            raise TypeError("weights was set to a non-sequence.")
    
    @property
    def total_weight(self):
        """
        Property storing the sum of all of the weights given to this
        distribution.
        """
        if not hasattr(self, '_total_weight'):
            self._total_weight = np.sum(self.weights)
        return self._total_weight
    
    @property
    def discrete_cdf_values(self):
        """
        Property storing the upper bound of CDF values allocated to each model.
        The sequence is monotonically increasing and its last element is equal
        to 1.
        """
        if not hasattr(self, '_discrete_cdf_values'):
            self._discrete_cdf_values =\
                np.cumsum(self.weights) / self.total_weight
        return self._discrete_cdf_values
    
    def draw(self, source, shape=None, random=rand):
        """
        Draws a point (or more) from the jumping_distribution.
        
        source: the source point of the jump. must be either a single number in
                the 1D case or a 1D vector in the multi-dimensional case
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
        if shape is None:
            uniform = random.uniform()
            idistribution = np.argmax(self.discrete_cdf_values > uniform)
            jumping_distribution = self.jumping_distributions[idistribution]
            return jumping_distribution.draw(source, random=random)
        else:
            if type(shape) in int_types:
                shape = (shape,)
            size = np.prod(shape)
            uniforms = random.uniform(size=size)
            if self.numparams == 1:
                intermediate_shape = (size,)
                final_shape = shape
            else:
                intermediate_shape = (size, self.numparams)
                final_shape = shape + (self.numparams)
            draws = np.ndarray(intermediate_shape)
            cdf_upper_bound = 0
            for (idistribution, jumping_distribution) in\
                enumerate(self.jumping_distributions):
                cdf_lower_bound = cdf_upper_bound
                cdf_upper_bound = self.discrete_cdf_values[idistribution]
                relevant_indices = np.nonzero((uniforms < cdf_upper_bound) &\
                    (uniforms >= cdf_lower_bound))[0]
                ndraw = len(relevant_indices)
                draws[relevant_indices,...] = jumping_distribution.draw(\
                    source, shape=ndraw, random=random)
            return np.reshape(draws, final_shape)
    
    def log_value(self, source, destination):
        """
        Computes the logarithm of the value of this distribution at the given
        point. It must be implemented by all subclasses.
        
        source, destination: either single values (if distribution is 1D) or
                             arrays of values
        
        returns: single number, logarithm of value of this distribution at the
                 given point
        """
        values = np.exp([jumping_distribution.log_value(source, destination)\
            for jumping_distribution in self.jumping_distributions])
        return np.log(np.sum(values * self.weights) / self.total_weight)
    
    @property
    def numparams(self):
        """
        Property storing the integer number of parameters described by this
        distribution. It must be implemented by all subclasses.
        """
        if not hasattr(self, '_numparams'):
            raise AttributeError("numparams was referenced before " +\
                "distributions was set.")
        return self._numparams
    
    def __eq__(self, other):
        """
        Tests for equality between this distribution and other. All subclasses
        must implement this function.
        
        other: JumpingDistribution with which to check for equality
        
        returns: True or False
        """
        if isinstance(other, JumpingDistributionSum):
            if self.num_jumping_distributions == other.num_jumping_distributions:
                distributions_equal = all([(sdistribution == odistribution)\
                    for (sdistribution, odistribution) in zip(\
                    self.jumping_distributions, other.jumping_distributions)])
                weights_equal =\
                    np.allclose(self.weights, other.weights, rtol=1e-6, atol=0)
                return distributions_equal and weights_equal
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
        if not hasattr(self, '_is_discrete'):
            raise AttributeError("is_discrete was referenced before " +\
                "jumping_distributions was set.")
        return self._is_discrete
    
    def fill_hdf5_group(self, group, save_metadata=True):
        """
        Fills the given hdf5 file group with information about this
        distribution. All subclasses must implement this function.
        
        group: hdf5 file group to fill with information about this distribution
        save_metadata: if True, attempts to save metadata alongside
                                distribution and throws error if it fails
                       if False, metadata is ignored in saving process
        """
        group.attrs['class'] = 'JumpingDistributionSum'
        group.attrs['num_jumping_distributions'] =\
            self.num_jumping_distributions
        subgroup = group.create_group('jumping_distributions')
        for (idistribution, jumping_distribution) in\
            enumerate(self.jumping_distributions):
            jumping_distribution.fill_hdf5_group(\
                subgroup.create_group('{:d}'.format(idistribution)))
        group.create_dataset('weights', data=self.weights)
    
    @staticmethod
    def load_from_hdf5_group(group, *jumping_distribution_classes):
        """
        Loads a JumpingDistribution from the given hdf5 file group.
        
        group: the same hdf5 file group which fill_hdf5_group was called on
               when this Distribution was saved
        jumping_distribution_classes: sequence (of length
                                      num_jumping_distributions) of class
                                      objects which can be used to load the
                                      sub-distributions
        
        returns: a JumpingDistribution object created from the information in
                 the given group
        """
        try:
            assert(group.attrs['class'] == 'JumpingDistributionSum')
            assert(group.attrs['num_jumping_distributions'] ==\
                len(jumping_distribution_classes))
        except:
            raise ValueError("The given group does not appear to contain a " +\
                "JumpingDistributionSum object.")
        weights = group['weights'].value
        subgroup = group['jumping_distributions']
        jumping_distributions = []
        for (icls, cls) in enumerate(jumping_distribution_classes):
            subsubgroup = subgroup['{:d}'.format(icls)]
            jumping_distribution = cls.load_from_hdf5_group(subsubgroup)
            jumping_distributions.append(jumping_distribution)
        return JumpingDistributionSum(jumping_distributions, weights)

