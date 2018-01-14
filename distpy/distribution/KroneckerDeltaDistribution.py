"""
File: distpy/Distribution.py
Author: Keith Tauscher
Date: 6 Aug 2017

Description: File containing class representing distribution which .
"""
from .Distribution import Distribution

class KroneckerDeltaDistribution(Distribution):
    """
    Distribution which always returns the same discrete value.
    """
    def __init__(self, value):
        """
        Initializes a KroneckerDeltaDistribution class
        
        value: value which is always returned by this distribution
        """
        self.value = value
    
    @property
    def value(self):
        """
        Property storing the value which is always returned by this
        distribution.
        """
        if not hasattr(self, '_value'):
            raise AttributeError("value referenced before it wass set.")
        return self._value
    
    @value.setter
    def value(self, value):
        """
        Setter for the value which is always returned by this distribution
        
        value: value which is always returned by this distribution
        """
        if type(value) in numerical_types:
            self._value = value
        else:
            raise TypeError("value was set to a non-number.")
    
    def draw(self, shape=None):
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
        
        returns: either single value (if distribution is 1D) or array of values
        """
        if shape is None:
            return self.value
        else:
            return self.value * np.ones(shape)
    
    def log_value(self, point):
        """
        Computes the logarithm of the value of this distribution at the given
        point. It must be implemented by all subclasses.
        
        point: single value
        
        returns: single number, logarithm of value of this distribution at the
                 given point
        """
        if point == self.value:
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
        return 1
    
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
            return self.value == other.value
        else:
            return False
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with information about this
        distribution. All subclasses must implement this function.
        
        group: hdf5 file group to fill with information about this distribution
        """
        group.attrs['class'] = 'KroneckerDeltaDistribution'
        group.attrs['value'] = self.value
    
    @property
    def confidence_interval(self):
        """
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

