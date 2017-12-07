"""
File: distpy/UniformDistribution.py
Author: Keith Tauscher
Date: 6 Aug 2017

Description: File containing a class representing a uniform distribution.
"""
import numpy as np
import numpy.random as rand
from ..util import numerical_types
from .Distribution import Distribution

class UniformDistribution(Distribution):
    """
    Class representing a uniform distribution. Uniform distributions are
    the least informative possible distributions (on a given
    support) and are thus ideal when ignorance abound.
    """
    def __init__(self, low=0., high=1.):
        """
        Creates a new UniformDistribution with the given range.
        
        low lower limit of pdf (defaults to 0)
        high upper limit of pdf (defaults to 1)
        """
        if (type(low) in numerical_types) and (type(high) in numerical_types):
            if low < high:
                self.low = low
                self.high = high
            elif high < low:
                self.low = high
                self.high = low
            else:
                raise ValueError('The high and low endpoints of a ' +\
                    'UniformDistribution are equal!')
        else:
            raise ValueError('Either the low or high endpoint of a ' +\
                'UniformDistribution was not of a numerical type.')
        self._log_P = - np.log(self.high - self.low)

    @property
    def numparams(self):
        """
        Only univariate uniform distributions are included here so numparams
        always returns 1.
        """
        return 1

    def draw(self, shape=None):
        """
        Draws and returns a value from this distribution using numpy.random.
        
        shape: if None, returns single random variate
                        (scalar for univariate ; 1D array for multivariate)
               if int, n, returns n random variates
                          (1D array for univariate ; 2D array for multivariate)
               if tuple of n ints, returns that many random variates
                                   n-D array for univariate ;
                                   (n+1)-D array for multivariate
        """
        return rand.uniform(low=self.low, high=self.high, size=shape)


    def log_value(self, point):
        """
        Evaluates and returns the log of the value of this distribution when
        the variable is value.
        
        point: numerical value of the variable
        """
        if (point >= self.low) and (point <= self.high):
            return self._log_P
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
            return np.allclose([self.low, self.high], [other.low, other.high],\
                rtol=0, atol=1e-9)
        else:
            return False
    
    def inverse_cdf(self, cdf):
        """
        Inverse of cumulative distribution function.
        
        cdf: value between 0 and 1
        """
        return (self.low + ((self.high - self.low) * cdf))
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data from this distribution. All
        that needs to be saved is the class name and high and low values.
        
        group: hdf5 file group to fill
        """
        group.attrs['class'] = 'UniformDistribution'
        group.attrs['low'] = self.low
        group.attrs['high'] = self.high
    
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

