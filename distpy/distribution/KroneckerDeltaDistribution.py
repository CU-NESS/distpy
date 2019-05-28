"""
File: distpy/distribution/KroneckerDeltaDistribution.py
Author: Keith Tauscher
Date: 12 Feb 2018

Description: File containing class representing distribution which always takes
             the same value.
"""
import numpy as np
from ..util import int_types, numerical_types, sequence_types, bool_types
from .Distribution import Distribution

class KroneckerDeltaDistribution(Distribution):
    """
    Distribution which always returns the same discrete value.
    """
    def __init__(self, value, is_discrete=True, metadata=None):
        """
        Initializes a KroneckerDeltaDistribution class
        
        value: value which is always returned by this distribution
        is_discrete: True if the variable underlying this distribution is
                     discrete. False otherwise (default True)
        metadata: data to store alongside this distribution
        """
        self.value = value
        self.is_discrete = is_discrete
        self.metadata = metadata
    
    @property
    def value(self):
        """
        Property storing the value which is always returned by this
        distribution.
        """
        if not hasattr(self, '_value'):
            raise AttributeError("value referenced before it was set.")
        return self._value
    
    @value.setter
    def value(self, value):
        """
        Setter for the value which is always returned by this distribution
        
        value: value which is always returned by this distribution
        """
        if type(value) in numerical_types:
            self._value = np.array([value])
        elif type(value) in sequence_types:
            value = np.array(value)
            if (value.ndim == 1) and (value.size > 0):
                self._value = value
            else:
                raise ValueError("KroneckerDeltaDistribution must be " +\
                    "initialized with either a number value or a non-empty " +\
                    "1D numpy.ndarray value.")
        else:
            raise TypeError("value was set to a non-number.")
    
    def draw(self, shape=None, random=None):
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
        
        if type(shape) is type(None):
            return_value = self.value
        else:
            if type(shape) in int_types:
                shape = (shape,)
            return_value = self.value * np.ones(shape + (self.numparams,))
        if self.numparams == 1:
            return return_value[...,0]
        else:
            return return_value
    
    def log_value(self, point):
        """
        Computes the logarithm of the value of this distribution at the given
        point. It must be implemented by all subclasses.
        
        point: single value
        
        returns: single number, logarithm of value of this distribution at the
                 given point
        """
        if np.all(point == self.value):
            return 0.
        else:
            return -np.inf
    
    @property
    def gradient_computable(self):
        """
        Property which stores whether the gradient of the given distribution
        has been implemented.
        """
        return True
    
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
        return (point * 0)
    
    @property
    def hessian_computable(self):
        """
        Property which stores whether the hessian of the given distribution
        has been implemented.
        """
        return True
    
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
        return (point * 0)
    
    @property
    def numparams(self):
        """
        Property storing the integer number of parameters described by this
        distribution. It must be implemented by all subclasses.
        """
        if not hasattr(self, '_numparams'):
            self._numparams = len(self.value)
        return self._numparams
    
    def to_string(self):
        """
        Returns a string representation of this distribution. It must be
        implemented by all subclasses.
        """
        return 'KroneckerDeltaDistribution({!s})'.format(self.value)
    
    def __eq__(self, other):
        """
        Tests for equality between this distribution and other. All subclasses
        must implement this function.
        
        other: Distribution with which to check for equality
        
        returns: True or False
        """
        if isinstance(other, KroneckerDeltaDistribution):
            value_equal = np.all(self.value == other.value)
            metadata_equal = self.metadata_equal(other)
            return (value_equal and metadata_equal)
        else:
            return False
    
    @property
    def is_discrete(self):
        """
        Property storing a boolean describing whether this distribution is
        discrete (True) or continuous (False).
        """
        if not hasattr(self, '_is_discrete'):
            raise AttributeError("is_discrete referenced before it was set.")
        return self._is_discrete
    
    @property
    def minimum(self):
        """
        Property storing the minimum allowable value(s) in this distribution.
        """
        return self.value
    
    @property
    def maximum(self):
        """
        Property storing the maximum allowable value(s) in this distribution.
        """
        return self.value
    
    @is_discrete.setter
    def is_discrete(self, value):
        """
        Setter for whether this distribution is discrete or continuous (the
        form itself does not determine this since this distribution cannot be
        drawn from).
        
        value: must be a bool (True for discrete, False for continuous)
        """
        if type(value) in bool_types:
            self._is_discrete = value
        else:
            raise TypeError("is_discrete was set to a non-bool.")
    
    def fill_hdf5_group(self, group, save_metadata=True):
        """
        Fills the given hdf5 file group with information about this
        distribution. All subclasses must implement this function.
        
        group: hdf5 file group to fill with information about this distribution
        save_metadata: if True, attempts to save metadata alongside
                                distribution and throws error if it fails
                       if False, metadata is ignored in saving process
        """
        group.attrs['class'] = 'KroneckerDeltaDistribution'
        group.attrs['value'] = self.value
        if save_metadata:
            self.save_metadata(group)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a KroneckerDeltaDistribution from the given hdf5 file group.
        
        group: the same hdf5 file group which fill_hdf5_group was called on
               when this Distribution was saved
        
        returns: a KroneckerDeltaDistribution object created from the
                 information in the given group
        """
        try:
            assert group.attrs['class'] == 'KroneckerDeltaDistribution'
        except:
            raise TypeError("The given hdf5 file doesn't seem to contain a " +\
                "KroneckerDeltaDistribution.")
        metadata = Distribution.load_metadata(group)
        value = group.attrs['value']
        return KroneckerDeltaDistribution(value, metadata=metadata)
    
    @property
    def confidence_interval(self):
        """
        All confidence intervals of the KroneckerDelta distribution are
        infinitely small and centered on the value of the distribution.
        
        returns: (value, value) where value is the peak of this distribution
        """
        return (self.value, self.value)
    
    def left_confidence_interval(self, probability_level):
        """
        Finds confidence interval furthest to the left.
        
        probability_level: the probability with which a random variable with
                           this distribution will exist in returned interval
        
        returns: (low, high) interval
        """
        return self.confidence_interval
    
    def central_confidence_interval(self, probability_level):
        """
        Finds confidence interval which has same probability of lying above or
        below interval.
        
        probability_level: the probability with which a random variable with
                           this distribution will exist in returned interval
        
        returns: (low, high) interval
        """
        return self.confidence_interval
    
    def right_confidence_interval(self, probability_level):
        """
        Finds confidence interval furthest to the right.
        
        probability_level: the probability with which a random variable with
                           this distribution will exist in returned interval
        
        returns: (low, high) interval
        """
        return self.confidence_interval
    
    def copy(self):
        """
        Returns a deep copy of this Distribution. This function ignores
        metadata.
        """
        return KroneckerDeltaDistribution(self.value,\
            is_discrete=self.is_discrete)

