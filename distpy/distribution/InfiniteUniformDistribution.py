"""
File: distpy/distribution/InfiniteUniformDistribution.py
Author: Keith Tauscher
Date: 12 Feb 2018

Description: File containing class representing an improper uniform
             "distribution". This Distribution cannot be drawn from as there is
             zero probability of its variate appearing in any given finite
             interval.
"""
from ..util import int_types, bool_types
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
    def __init__(self, ndim, metadata=None, is_discrete=False):
        """
        Initializes a new InfiniteUniformDistribution
        
        ndim: the dimension of this distribution
        metadata: data to store alongside this distribution
        is_discrete: True if the variable underlying this distribution is
                     discrete. False otherwise (default False)
        """
        self.numparams = ndim
        self.is_discrete = is_discrete
        self.metadata = metadata
    
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
        if isinstance(other, InfiniteUniformDistribution):
            return self.metadata_equal(other)
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
        group.attrs['class'] = 'InfiniteUniformDistribution'
        group.attrs['ndim'] = self.numparams
        if save_metadata:
            self.save_metadata(group)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads an InfiniteUniformDistribution from the given hdf5 file group.
        
        group: the same hdf5 file group which fill_hdf5_group was called on
               when this Distribution was saved
        
        returns: an InfiniteUniformDistribution object created from the
                 information in the given group
        """
        try:
            assert group.attrs['class'] == 'InfiniteUniformDistribution'
        except:
            raise TypeError("The given hdf5 file doesn't seem to contain a " +\
                "InfiniteUniformDistribution.")
        metadata = Distribution.load_metadata(group)
        ndim = group.attrs['ndim']
        return InfiniteUniformDistribution(ndim, metadata=metadata)
    
    @property
    def can_give_confidence_intervals(self):
        """
        Confidence intervals for most distributions can be generated as long as
        this distribution describes only one dimension.
        """
        return False

