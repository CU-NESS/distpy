"""
File: distpy/Distribution.py
Author: Keith Tauscher
Date: 6 Aug 2017

Description: File containing base class for all distributions.
"""
from ..util import Savable

def raise_cannot_instantiate_distribution_error():
    raise NotImplementedError("Some part of Distribution class was not " +\
        "implemented by subclass or Distribution is being instantiated.")

class Distribution(Savable):
    """
    This class exists for error catching. Since it exists as
    a superclass of all the distributions, one can call
    isinstance(to_check, Distribution) to see if to_check is indeed a kind of
    distribution.
    
    All subclasses of this one will implement
    self.draw() --- draws a point from this distribution
    self.log_value(point) --- computes the log of the value of this
                              distribution at the given point
    self.numparams --- property, not function, storing number of parameters
    self.to_string() --- string summary of this distribution
    self.__eq__(other) --- checks for equality with another object
    self.fill_hdf5_group(group) --- fills given hdf5 group with data from
                                    distribution
    
    In draw() and log_value(), point is a configuration. It could be a
    single number for a univariate distribution or a numpy.ndarray for a
    multivariate distribution.
    """
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
        raise_cannot_instantiate_distribution_error()
    
    def log_value(self, point):
        """
        Computes the logarithm of the value of this distribution at the given
        point. It must be implemented by all subclasses.
        
        point: either single value (if distribution is 1D) or array of values
        
        returns: single number, logarithm of value of this distribution at the
                 given point
        """
        raise_cannot_instantiate_distribution_error()
    
    def __call__(self, point):
        """
        Alias for log_value function.
        
        point: either single value (if distribution is 1D) or array of values
        
        returns: single number, logarithm of value of this distribution at the
                 given point
        """
        return self.log_value(point)
    
    @property
    def numparams(self):
        """
        Property storing the integer number of parameters described by this
        distribution. It must be implemented by all subclasses.
        """
        raise_cannot_instantiate_distribution_error()
    
    def to_string(self):
        """
        Returns a string representation of this distribution. It must be
        implemented by all subclasses.
        """
        raise_cannot_instantiate_distribution_error()
    
    def __eq__(self, other):
        """
        Tests for equality between this distribution and other. All subclasses
        must implement this function.
        
        other: Distribution with which to check for equality
        
        returns: True or False
        """
        raise_cannot_instantiate_distribution_error()
    
    def fill_hdf5_group(group):
        """
        Fills the given hdf5 file group with information about this
        distribution. All subclasses must implement this function.
        
        group: hdf5 file group to fill with information about this distribution
        """
        raise_cannot_instantiate_distribution_error()
    
    def __ne__(self, other):
        """
        This merely enforces that (a!=b) equals (not (a==b)) for all
        distribution objects a and b.
        """
        return (not self.__eq__(other))
    
    @property
    def can_give_confidence_intervals(self):
        """
        Confidence intervals for most distributions can be generated as long as
        this distribution describes only one dimension.
        """
        return (self.numparams == 1)
    
    def left_confidence_interval(self, probability_level):
        """
        Finds confidence interval furthest to the left.
        
        probability_level: the probability with which a random variable with
                           this distribution will exist in returned interval
        
        returns: (low, high) interval
        """
        if self.can_give_confidence_intervals:
            return (self.inverse_cdf(0), self.inverse_cdf(probability_level))
        else:
            raise ValueError("Confidence intervals cannot be found for " +\
                "this distribution.")
    
    def central_confidence_interval(self, probability_level):
        """
        Finds confidence interval which has same probability of lying above or
        below interval.
        
        probability_level: the probability with which a random variable with
                           this distribution will exist in returned interval
        
        returns: (low, high) interval
        """
        if self.numparams == 1:
            return (self.inverse_cdf((1 - probability_level) / 2),\
                self.inverse_cdf((1 + probability_level) / 2))
        else:
            raise ValueError("Confidence intervals cannot be found for " +\
                "this distribution.")
    
    def right_confidence_interval(self, probability_level):
        """
        Finds confidence interval furthest to the right.
        
        probability_level: the probability with which a random variable with
                           this distribution will exist in returned interval
        
        returns: (low, high) interval
        """
        if self.numparams == 1:
            return\
                (self.inverse_cdf(1 - probability_level), self.inverse_cdf(1))
        else:
            raise ValueError("Confidence intervals cannot be found for " +\
                "this distribution.")

