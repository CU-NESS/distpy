"""
File: distpy/distribution/DistributionSum.py
Author: Keith Tauscher
Date: 10 Jun 2018

Description: File containing class which represents a weighted sum of
             distributions.
"""
import numpy as np
import numpy.random as rand
from ..util import int_types, sequence_types
from .Distribution import Distribution

class DistributionSum(Distribution):
    """
    Class which represents a weighted sum of distributions.
    """
    def __init__(self, distributions, weights, metadata=None):
        """
        Creates a new DistributionSum object out of the given distributions.
        
        distributions: sequence of Distribution objects with the same numparams
        weights: sequence of numbers with which to combine (need not be
                 normalized but they must all be positive)
        """
        self.distributions = distributions
        self.weights = weights
        self.metadata = metadata
    
    @property
    def distributions(self):
        """
        Property storing a list of Distribution objects which make up this
        DistributionSum.
        """
        if not hasattr(self, '_distributions'):
            raise AttributeError("distributions was referenced before it " +\
                "was set.")
        return self._distributions
    
    @distributions.setter
    def distributions(self, value):
        """
        Setter for the Distribution objects making up this DistributionSum.
        
        value: sequence of Distributions making up this DistributionSum
        """
        if type(value) in sequence_types:
            if all([isinstance(element, Distribution) for element in value]):
                unique_nums_of_parameters =\
                    np.unique([element.numparams for element in value])
                unique_is_discretes =\
                    np.unique([element.is_discrete for element in value])
                if (len(unique_nums_of_parameters) == 1) and\
                    (len(unique_is_discretes) == 1):
                    self._distributions = [element for element in value]
                    self._numparams = unique_nums_of_parameters[0]
                    self._is_discrete = unique_is_discretes[0]
                else:
                    raise ValueError("Not all distributions added together " +\
                        "here had the same number of parameters.")
            else:
                raise TypeError("At least one element of the distributions " +\
                    "sequence was not a Distribution object.")
        else:
            raise TypeError("distributions was set to a non-sequence.")
    
    @property
    def num_distributions(self):
        """
        Property storing the number of Distribution objects which make up this
        DistributionSum object.
        """
        if not hasattr(self, '_num_distributions'):
            self._num_distributions = len(self.distributions)
        return self._num_distributions
    
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
            if len(value) == self.num_distributions:
                value = np.array(value)
                if np.all(np.zeros(self.num_distributions) < value):
                    self._weights = value
                else:
                    raise ValueError("At least one weight was non-positive.")
            else:
                raise ValueError("The length of the weights sequence was " +\
                    "not the same as the length of the distributions " +\
                    "sequence.")
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
    
    def draw(self, shape=None, random=rand):
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
        if shape is None:
            uniform = random.uniform()
            idistribution = np.argmax(self.discrete_cdf_values > uniform)
            distribution = self.distributions[idistribution]
            return distribution.draw(random=random)
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
            for (idistribution, distribution) in enumerate(self.distributions):
                cdf_lower_bound = cdf_upper_bound
                cdf_upper_bound = self.discrete_cdf_values[idistribution]
                relevant_indices = np.nonzero((uniforms < cdf_upper_bound) &\
                    (uniforms >= cdf_lower_bound))[0]
                ndraw = len(relevant_indices)
                draws[relevant_indices,...] =\
                    distribution.draw(shape=ndraw, random=random)
            return np.reshape(draws, final_shape)
    
    def log_value(self, point):
        """
        Computes the logarithm of the value of this distribution at the given
        point. It must be implemented by all subclasses.
        
        point: either single value (if distribution is 1D) or array of values
        
        returns: single number, logarithm of value of this distribution at the
                 given point
        """
        values = np.exp([distribution.log_value(point)\
            for distribution in self.distributions])
        return np.log(np.sum(values * self.weights) / self.total_weight)
    
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
        distribution. It must be implemented by all subclasses.
        """
        if not hasattr(self, '_numparams'):
            raise AttributeError("numparams was referenced before " +\
                "distributions was set.")
        return self._numparams
    
    def to_string(self):
        """
        Returns a string representation of this distribution. It must be
        implemented by all subclasses.
        """
        return "Sum of {:d} dists".format(self.num_distributions)
    
    def __eq__(self, other):
        """
        Tests for equality between this distribution and other. All subclasses
        must implement this function.
        
        other: Distribution with which to check for equality
        
        returns: True or False
        """
        if isinstance(other, DistributionSum):
            if self.num_distributions == other.num_distributions:
                distributions_equal = all([(sdistribution == odistribution)\
                    for (sdistribution, odistribution) in\
                    zip(self.distributions, other.distributions)])
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
                "distributions was set.")
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
        group.attrs['class'] = 'DistributionSum'
        group.attrs['num_distributions'] = self.num_distributions
        subgroup = group.create_group('distributions')
        for (idistribution, distribution) in enumerate(self.distributions):
            distribution.fill_hdf5_group(\
                subgroup.create_group('{:d}'.format(idistribution)))
        group.create_dataset('weights', data=self.weights)
        if save_metadata:
            self.save_metadata(group)
    
    @staticmethod
    def load_from_hdf5_group(group, *distribution_classes):
        """
        Loads a Distribution from the given hdf5 file group. All Distribution
        subclasses must implement this method if things are to be saved in hdf5
        files.
        
        group: the same hdf5 file group which fill_hdf5_group was called on
               when this Distribution was saved
        distribution_classes: sequence (of length num_distributions) of class
                              objects which can be used to load the
                              sub-distributions
        
        returns: a Distribution object created from the information in the
                 given group
        """
        try:
            assert(group.attrs['class'] == 'DistributionSum')
            assert(\
                group.attrs['num_distributions'] == len(distribution_classes))
        except:
            raise ValueError("The given group does not appear to contain a " +\
                "DistributionSum object.")
        metadata = Distribution.load_metadata(group)
        weights = group['weights'][()]
        subgroup = group['distributions']
        distributions = []
        for (icls, cls) in enumerate(distribution_classes):
            subsubgroup = subgroup['{:d}'.format(icls)]
            distribution = cls.load_from_hdf5_group(subsubgroup)
            distributions.append(distribution)
        return DistributionSum(distributions, weights, metadata=metadata)
    
    @property
    def can_give_confidence_intervals(self):
        """
        Confidence intervals for most distributions can be generated as long as
        this distribution describes only one dimension.
        """
        return False

