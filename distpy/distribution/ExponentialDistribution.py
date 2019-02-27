"""
File: distpy/distribution/ExponentialDistribution.py
Author: Keith Tauscher
Date: 12 Feb 2018

Description: File containing class representing an exponential distribution.
"""
from __future__ import division
import numpy as np
import numpy.random as rand
from ..util import numerical_types
from .Distribution import Distribution

class ExponentialDistribution(Distribution):
    """
    Distribution with exponential distribution. Exponential distributions are
    ideal for parameters which are naturally non-negative (or, if shift is
    used, are naturally above some certain lower cutoff)
    """
    def __init__(self, rate, shift=0., metadata=None):
        """
        Initializes a new ExponentialDistribution with the given parameters.
        
        rate the rate parameter of the distribution (number multiplied by x in
             exponent of pdf) (must be greater than 0)
        shift lower limit of the support of the distribution (defaults to 0)
        """
        self.rate = rate
        self.shift = shift
        self.metadata = metadata
    
    @property
    def rate(self):
        """
        Property storing the rate parameter (often denoted lambda) of this
        distribution.
        """
        if not hasattr(self, '_rate'):
            raise AttributeError("rate was referenced before it was set.")
        return self._rate
    
    @rate.setter
    def rate(self, value):
        """
        Setter for the rate of this distribution.
        
        value: single positive number
        """
        if type(value) in numerical_types:
            if value > 0:
                self._rate = value
            else:
                raise ValueError("rate property of ExponentialDistribution " +\
                    "was set to a non-positive number.")
        else:
            raise TypeError("rate property of ExponentialDistribution was " +\
                "set to a non-number.")
    
    @property
    def shift(self):
        """
        Property storing the shift parameter (i.e. the left endpoint of this
        distribution's domain of support).
        """
        if not hasattr(self, '_shift'):
            raise AttributeError("shift was referenced before it was set.")
        return self._shift
    
    @shift.setter
    def shift(self, value):
        """
        Setter for the shift of this distribution.
        
        value: single real number
        """
        if type(value) in numerical_types:
            self._shift = value
        else:
            raise TypeError("shift property of ExponentialDistribution was " +\
                "set to a non-number.")
    
    @staticmethod
    def create_from_mean_and_variance(mean, variance, metadata=None):
        """
        Creates a new ExponentialDistribution from the mean and variance
        instead of the rate and shift parameters.
        
        mean: number equal to the mean of the desired distribution
        variance: number equal to the variance of the desired distribution
        metadata: data to be stored alongside distribution, should be hdf5-able
        """
        standard_deviation = np.sqrt(variance)
        rate = (1 / standard_deviation)
        shift = (mean - standard_deviation)
        return ExponentialDistribution(rate, shift=shift, metadata=None)
    
    @property
    def numparams(self):
        """
        Exponential distribution pdf is univariate so numparams always returns
        1.
        """
        return 1
    
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
        return\
            random.exponential(scale=(1 / self.rate), size=shape) + self.shift
    
    def log_value(self, point):
        """
        Evaluates and returns the log of the value of this distribution when
        the variable is point.
        
        point: numerical value of the variable
        """
        val_min_shift = point - self.shift
        if val_min_shift < 0:
            return -np.inf
        return (np.log(self.rate) - (self.rate * val_min_shift))

    def to_string(self):
        """
        Finds and returns a string version of this ExponentialDistribution.
        """
        if self.shift != 0:
            return "Exp({0:.2g}, shift={1:.2g})".format(self.rate, self.shift)
        return "Exp({:.2g})".format(self.rate)
    
    def __eq__(self, other):
        """
        Checks for equality of this distribution with other. Returns True if
        other is an ExponentialDistribution with the same rate (to 10^9 dynamic
        range) and shift (down to 1e-9) and False otherwise.
        """
        if isinstance(other, ExponentialDistribution):
            rate_close = np.isclose(self.rate, other.rate, rtol=1e-9, atol=0)
            shift_close =\
                np.isclose(self.shift, other.shift, rtol=0, atol=1e-9)
            metadata_equal = self.metadata_equal(other)
            return all([rate_close, shift_close, metadata_equal])
        else:
            return False
    
    def inverse_cdf(self, cdf):
        """
        Inverse of the cumulative distribution function.
        
        cdf: value between 0 and 1
        """
        return (self.shift - (np.log(1 - cdf) / self.rate))
    
    @property
    def minimum(self):
        """
        Property storing the minimum allowable value(s) in this distribution.
        """
        return 0
    
    @property
    def maximum(self):
        """
        Property storing the maximum allowable value(s) in this distribution.
        """
        return None
    
    @property
    def is_discrete(self):
        """
        Property storing a boolean describing whether this distribution is
        discrete (True) or continuous (False).
        """
        return False
    
    def fill_hdf5_group(self, group, save_metadata=True):
        """
        Fills the given hdf5 file group with data about this distribution. The
        only things to save are the class name, rate, and shift.
        
        group: hdf5 file group to fill
        save_metadata: if True, attempts to save metadata alongside
                                distribution and throws error if it fails
                       if False, metadata is ignored in saving process
        """
        group.attrs['class'] = 'ExponentialDistribution'
        group.attrs['rate'] = self.rate
        group.attrs['shift'] = self.shift
        if save_metadata:
            self.save_metadata(group)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads an ExponentialDistribution from the given hdf5 file group.
        
        group: the same hdf5 file group which fill_hdf5_group was called on
               when this Distribution was saved
        
        returns: an ExponentialDistribution object created from the information
                 in the given group
        """
        try:
            assert group.attrs['class'] == 'ExponentialDistribution'
        except:
            raise TypeError("The given hdf5 file doesn't seem to contain a " +\
                "ExponentialDistribution.")
        metadata = Distribution.load_metadata(group)
        rate = group.attrs['rate']
        shift = group.attrs['shift']
        return ExponentialDistribution(rate, shift=shift, metadata=metadata)
    
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
        return -self.rate
    
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
        return ExponentialDistribution(self.rate, self.shift)

