"""
File: distpy/distribution/InfiniteUniformDistribution.py
Author: Keith Tauscher
Date: 25 Mar 2019

Description: File containing class representing an improper uniform
             "distribution". This Distribution cannot be drawn from as there is
             zero probability of its variate appearing in any given finite
             interval.
"""
import numpy as np
from ..util import int_types, bool_types, numerical_types, sequence_types
from .Distribution import Distribution

class InfiniteUniformDistribution(Distribution):
    """
    A class representing a uniform distribution over all possible inputs (this
    is not a "proper" prior; it cannot be drawn from).
    """
    def __init__(self, ndim=1, minima=None, maxima=None,\
        is_discrete=False, metadata=None):
        """
        Initializes a new InfiniteUniformDistribution
        
        ndim: the dimension of this distribution, default 1
        minima: if None, no lower bound is used for any parameters
                if a number, that is the number used as a lower bound for all
                             parameters
                otherwise (only if numparams > 1), must be a sequence of length
                                                   numparams containing None's
                                                   and/or numbers containing
                                                   lower bounds
        maxima: if None, no upper bound is used for any parameters
                if a number, that is the number used as a upper bound for all
                             parameters
                otherwise (only if numparams > 1), must be a sequence of length
                                                   numparams containing None's
                                                   and/or numbers containing
                                                   upper bounds
        is_discrete: True if the variable underlying this distribution is
                     discrete. False otherwise (default False)
        metadata: data to store alongside this distribution
        """
        self.numparams = ndim
        self.minima = minima
        self.maxima = maxima
        self.is_discrete = is_discrete
        self.metadata = metadata
    
    @property
    def minima(self):
        """
        Property storing the minimum variable value(s) in an array.
        """
        if not hasattr(self, '_minima'):
            raise AttributeError("minima was referenced before it was set.")
        return self._minima
    
    @minima.setter
    def minima(self, value):
        """
        Setter for the minimum value(s) of this distribution.
        
        value: if None, no lower bound is used for any parameters
               if a number, that is the number used as a lower bound for all
                            parameters
               otherwise (only if numparams > 1), must be a sequence of length
                                                  numparams containing None's
                                                  and/or numbers containing
                                                  lower bounds
        """
        if self.numparams == 1:
            if type(value) is type(None):
                self._minima = -np.inf
            elif type(value) in numerical_types:
                self._minima = value
            else:
                raise TypeError("minima was set to neither None nor a number.")
        elif type(value) is type(None):
            self._minima = np.ones((self.numparams,)) * (-np.inf)
        elif type(value) in numerical_types:
            self._minima = np.ones((self.numparams,)) * value
        elif type(value) in sequence_types:
            if len(value) == self.numparams:
                self._minima = np.array([((-np.inf)\
                    if (type(element) is type(None)) else element)\
                    for element in value])
            else:
                raise ValueError("The sequence of minima given to an " +\
                    "InfiniteUniformDistribution object was not of length " +\
                    "ndim.")
        else:
            raise TypeError("minima was set to neither None nor a number " +\
                "or sequence.")
    
    @property
    def maxima(self):
        """
        Property storing the maximum variable value(s) in an array.
        """
        if not hasattr(self, '_maxima'):
            raise AttributeError("maxima was referenced before it was set.")
        return self._maxima
    
    @maxima.setter
    def maxima(self, value):
        """
        Setter for the maximum value(s) of this distribution.
        
        value: if None, no upper bound is used for any parameters
               if a number, that is the number used as a upper bound for all
                            parameters
               otherwise (only if numparams > 1), must be a sequence of length
                                                  numparams containing None's
                                                  and/or numbers containing
                                                  upper bounds
        """
        if self.numparams == 1:
            if type(value) is type(None):
                self._maxima = np.inf
            elif type(value) in numerical_types:
                self._maxima = value
            else:
                raise TypeError("maxima was set to neither None nor a number.")
        elif type(value) is type(None):
            self._maxima = np.ones((self.numparams,)) * (np.inf)
        elif type(value) in numerical_types:
            self._maxima = np.ones((self.numparams,)) * value
        elif type(value) in sequence_types:
            if len(value) == self.numparams:
                self._maxima = np.array([(np.inf\
                    if (type(element) is type(None)) else element)\
                    for element in value])
            else:
                raise ValueError("The sequence of maxima given to an " +\
                    "InfiniteUniformDistribution object was not of length " +\
                    "ndim.")
        else:
            raise TypeError("maxima was set to neither None nor a number " +\
                "or sequence.")
        if np.any(np.all(np.isfinite(np.array([self.minima, self._maxima])),\
            axis=0)):
            raise ValueError("At least one parameter of an " +\
                "InfiniteUniformDistribution had both upper and lower " +\
                "bounds, meaning it would be better for it to be " +\
                "encapsulated with a regular UniformDistribution instead " +\
                "of an InfiniteUniformDistribution.")
    
    @property
    def all_left_infinite(self):
        """
        Property storing whether all parameters of this distribution have no
        lower bound.
        """
        if not hasattr(self, '_all_left_infinite'):
            self._all_left_infinite = (not np.any(np.isfinite(self.minima)))
        return self._all_left_infinite
    
    @property
    def all_right_infinite(self):
        """
        Property storing whether all parameters of this distribution have no
        upper bound.
        """
        if not hasattr(self, '_all_right_infinite'):
            self._all_right_infinite = (not np.any(np.isfinite(self.maxima)))
        return self._all_right_infinite
    
    @property
    def all_doubly_infinite(self):
        """
        Property storing whether this distribution is doubly infinite in all of
        its parameters. In this case, no comparison is done when this
        distribution is called.
        """
        if not hasattr(self, '_all_doubly_infinite'):
            self._all_doubly_infinite =\
                (self.all_left_infinite and self.all_right_infinite)
        return self._all_doubly_infinite
    
    def draw(self, shape=None, random=None):
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
        if ((self.all_left_infinite or np.all(point > self.minima)) and\
            (self.all_right_infinite or np.all(point < self.maxima))):
            return 0.
        else:
            return (-np.inf)
    
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
        if not isinstance(other, InfiniteUniformDistribution):
            return False
        if self.is_discrete != other.is_discrete:
            return False
        if np.any(self.minima != other.minima):
            return False
        if np.any(self.maxima != other.maxima):
            return False
        return self.metadata_equal(other)
    
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
        group.attrs['is_discrete'] = self.is_discrete
        group.attrs['ndim'] = self.numparams
        group.attrs['minima'] = self.minima
        group.attrs['maxima'] = self.maxima
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
        if 'is_discrete' in group.attrs:
            is_discrete = group.attrs['is_discrete']
        else:
            is_discrete = False
        if 'minima' in group.attrs:
            minima = group.attrs['minima']
        else:
            minima = None
        if 'maxima' in group.attrs:
            maxima = group.attrs['maxima']
        else:
            maxima = None
        return InfiniteUniformDistribution(ndim, minima=minima, maxima=maxima,\
            is_discrete=is_discrete, metadata=metadata)
    
    @property
    def minimum(self):
        """
        Property storing the minimum allowable value(s) in this distribution.
        """
        return [(element if np.isfinite(element) else None)\
            for element in self.minima]
    
    @property
    def maximum(self):
        """
        Property storing the maximum allowable value(s) in this distribution.
        """
        return [(element if np.isfinite(element) else None)\
            for element in self.maxima]
    
    @property
    def can_give_confidence_intervals(self):
        """
        Confidence intervals for most distributions can be generated as long as
        this distribution describes only one dimension.
        """
        return False
    
    def copy(self):
        """
        Returns a deep copy of this Distribution. This function ignores
        metadata.
        """
        return InfiniteUniformDistribution(self.numparams,\
            minima=self.minima, maxima=self.maxima,\
            is_discrete=self.is_discrete)

