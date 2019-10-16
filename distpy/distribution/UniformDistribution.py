"""
File: distpy/distribution/UniformDistribution.py
Author: Keith Tauscher
Date: Oct 15 2019

Description: File containing a class representing a uniform distribution.
"""
from __future__ import division
import numpy as np
import numpy.random as rand
from ..util import numerical_types, sequence_types
from .Distribution import Distribution

class UniformDistribution(Distribution):
    """
    Class representing a uniform distribution. Uniform distributions are
    the least informative possible distributions (on a given
    support) and are thus ideal when ignorance abound.
    """
    def __init__(self, low=0., high=1., metadata=None):
        """
        Creates a new UniformDistribution with the given range.
        
        low lower limit of pdf (defaults to 0)
        high upper limit of pdf (defaults to 1)
        """
        self.bounds = (low, high)
        self.metadata = metadata
    
    @property
    def bounds(self):
        """
        Property storing the lower and upper bounds of this distribution in a
        tuple.
        """
        if not hasattr(self, '_bounds'):
            raise AttributeError("bounds was referenced before it was set.")
        return self._bounds
    
    @bounds.setter
    def bounds(self, value):
        """
        Setter for the lower and upper bounds of this distribution.
        
        value: tuple of form (lower_bound, upper_bound)
        """
        if type(value) in sequence_types:
            if len(value) == 2:
                if all([(type(element) in numerical_types)\
                    for element in value]):
                    if value[0] == value[1]:
                        raise ValueError("The lower bound and upper bound " +\
                            "were set to the same number.")
                    else:
                        self._bounds = (min(value), max(value))
                else:
                    raise TypeError("At least one element of bounds was " +\
                        "not a number.")
            else:
                raise ValueError("bounds was set to a sequence of a length " +\
                    "other than two.")
        else:
            raise TypeError("bounds was set to a non-sequence.")
    
    @property
    def low(self):
        """
        Property storing the lower bound of this distribution.
        """
        return self.bounds[0]
    
    @property
    def high(self):
        """
        Property storing the upper bound of this distribution.
        """
        return self.bounds[1]
    
    @property
    def log_probability(self):
        """
        Property storing the log of the probability density inside the domain.
        """
        if not hasattr(self, '_log_probability'):
            self._log_probability = ((-1) * np.log(self.high - self.low))
        return self._log_probability

    @property
    def numparams(self):
        """
        Only univariate uniform distributions are included here so numparams
        always returns 1.
        """
        return 1
    
    @property
    def mean(self):
        """
        Property storing the mean of this distribution.
        """
        if not hasattr(self, '_mean'):
            self._mean = (self.low + self.high) / 2
        return self._mean
    
    @property
    def variance(self):
        """
        Property storing the covariance of this distribution.
        """
        if not hasattr(self, '_variance'):
            self._variance = ((self.high - self.low) ** 2) / 12
        return self._variance

    def draw(self, shape=None, random=rand):
        """
        Draws and returns a value from this distribution using numpy.random.
        
        shape: if None, returns single random variate
                        (scalar for univariate ; 1D array for multivariate)
               if int, n, returns n random variates
                          (1D array for univariate ; 2D array for multivariate)
               if tuple of n ints, returns that many random variates
                                   n-D array for univariate ;
                                   (n+1)-D array for multivariate
        random: the random number generator to use (default: numpy.random)
        """
        return random.uniform(low=self.low, high=self.high, size=shape)


    def log_value(self, point):
        """
        Evaluates and returns the log of the value of this distribution when
        the variable is value.
        
        point: numerical value of the variable
        """
        if (point >= self.low) and (point <= self.high):
            return self.log_probability
        return -np.inf
    
    def to_string(self):
        """
        Finds and returns a string representation of this distribution.
        """
        return "Uniform({0:.2g}, {1:.2g})".format(self.low, self.high)
    
    def __eq__(self, other):
        """
        Checks for equality of this distribution with other. Returns True if
        other is a UniformDistribution with the same high and low (down to 1e-9
        level) and False otherwise.
        """
        if isinstance(other, UniformDistribution):
            tol_kwargs = {'rtol': 0., 'atol': 1e-9}
            low_close = np.isclose(self.low, other.low, **tol_kwargs)
            high_close = np.isclose(self.high, other.high, **tol_kwargs)
            metadata_equal = self.metadata_equal(other)
            return all([low_close, high_close, metadata_equal])
        else:
            return False
    
    def inverse_cdf(self, cdf):
        """
        Inverse of cumulative distribution function.
        
        cdf: value between 0 and 1
        """
        return (self.low + ((self.high - self.low) * cdf))
    
    @property
    def minimum(self):
        """
        Property storing the minimum allowable value(s) in this distribution.
        """
        return self.low
    
    @property
    def maximum(self):
        """
        Property storing the maximum allowable value(s) in this distribution.
        """
        return self.high
    
    @property
    def is_discrete(self):
        """
        Property storing a boolean describing whether this distribution is
        discrete (True) or continuous (False).
        """
        return False
    
    def fill_hdf5_group(self, group, save_metadata=True):
        """
        Fills the given hdf5 file group with data from this distribution. All
        that needs to be saved is the class name and high and low values.
        
        group: hdf5 file group to fill
        save_metadata: if True, attempts to save metadata alongside
                                distribution and throws error if it fails
                       if False, metadata is ignored in saving process
        """
        group.attrs['class'] = 'UniformDistribution'
        group.attrs['low'] = self.low
        group.attrs['high'] = self.high
        if save_metadata:
            self.save_metadata(group)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a UniformDistribution from the given hdf5 file group.
        
        group: the same hdf5 file group which fill_hdf5_group was called on
               when this Distribution was saved
        
        returns: UniformDistribution object created from the information in the
                 given group
        """
        try:
            assert group.attrs['class'] == 'UniformDistribution'
        except:
            raise TypeError("The given hdf5 file doesn't seem to contain a " +\
                "UniformDistribution.")
        metadata = Distribution.load_metadata(group)
        low = group.attrs['low']
        high = group.attrs['high']
        return UniformDistribution(low=low, high=high, metadata=metadata)
    
    @property
    def gradient_computable(self):
        """
        Property which stores whether the gradient of the given distribution
        has been implemented. Since it has been implemented, it returns True.
        """
        return True
    
    def gradient_of_log_value(self, point):
        """
        Computes the derivative of log_value(point) with respect to the
        parameter.
        
        point: single number at which to evaluate the derivative
        
        returns: returns single number representing derivative of log value
        """
        return 0.
    
    @property
    def hessian_computable(self):
        """
        Property which stores whether the hessian of the given distribution
        has been implemented. Since it has been implemented, it returns True.
        """
        return True
    
    def hessian_of_log_value(self, point):
        """
        Computes the second derivative of log_value(point) with respect to the
        parameter.
        
        point: single value
        
        returns: single number representing second derivative of log value
        """
        return 0.
    
    def copy(self):
        """
        Returns a deep copy of this Distribution. This function ignores
        metadata.
        """
        return UniformDistribution(self.low, self.high)

