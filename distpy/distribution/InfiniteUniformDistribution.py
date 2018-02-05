"""
File: distpy/Distribution.py
Author: Keith Tauscher
Date: 6 Aug 2017

Description: File containing class representing an improper uniform
             "distribution". This Distribution cannot be drawn from as there is
             zero probability of its variate appearing in any given finite
             interval.
"""
from ..util import int_types
from .Distribution import Distribution

class InfiniteUniformDistribution(Distribution):
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
    def __init__(self, ndim):
        """
        Initializes a new InfiniteUniformDistribution
        
        ndim: the dimension of this distribution
        """
        self.numparams = ndim
    
    def draw(self, shape=None):
        """
        Draws a point from the distribution. Since this Distribution cannot be
        drawn from, this throws a NotImplementedError.
        """
        raise NotImplementedError("InfiniteUniformDistribution objects " +\
            "cannot be drawn from because there is zero probability of its " +\
            "variate appearing in any given finite interval.")
    
    def log_value(self, point):
        """
        Computes the logarithm of the value of this distribution at the given
        point. It must be implemented by all subclasses.
        
        point: either single value (if distribution is 1D) or array of values
        
        returns: single number, logarithm of value of this distribution at the
                 given point
        """
        return 0.
    
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
        return np.zeros(self.numparams)
    
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
        return np.zeros((self.numparams, self.numparams))
    
    @property
    def numparams(self):
        """
        Property storing the integer number of parameters described by this
        distribution.
        """
        if not hasattr(self, '_numparams'):
            raise AttributeError("numparams referenced before it was set.")
        return self._numparams
    
    @numparams.setter
    def numparams(self, value):
        """
        Setter for the dimension of this Distribution.
        
        value: a positive integer
        """
        if type(value) in int_types:
            if value > 0:
                self._numparams = value
            else:
                raise ValueError("numparams was set to a non-positive " +\
                    "integer.")
        else:
            raise TypeError("numparams was set to a non-integer.")
    
    def to_string(self):
        """
        Returns a string representation of this distribution. It must be
        implemented by all subclasses.
        """
        return 'InfiniteUniform'
    
    def __eq__(self, other):
        """
        Tests for equality between this distribution and other. All subclasses
        must implement this function.
        
        other: Distribution with which to check for equality
        
        returns: True or False
        """
        return isinstance(other, InfiniteUniformDistribution)
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with information about this
        distribution. All subclasses must implement this function.
        
        group: hdf5 file group to fill with information about this distribution
        """
        group.attrs['class'] = 'InfiniteUniformDistribution'
    
    @property
    def can_give_confidence_intervals(self):
        """
        Confidence intervals for most distributions can be generated as long as
        this distribution describes only one dimension.
        """
        return False

