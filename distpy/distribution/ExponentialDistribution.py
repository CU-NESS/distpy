"""
File: distpy/ExponentialDistribution.py
Author: Keith Tauscher
Date: 6 Aug 2017

Description: File containing class representing an exponential distribution.
"""
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
    def __init__(self, rate, shift=0.):
        """
        Initializes a new ExponentialDistribution with the given parameters.
        
        rate the rate parameter of the distribution (number multiplied by x in
             exponent of pdf) (must be greater than 0)
        shift lower limit of the support of the distribution (defaults to 0)
        """
        if type(rate) in numerical_types:
            if rate > 0:
                self.rate = (rate * 1.)
            else:
                raise ValueError('The rate parameter given to an ' +\
                    'ExponentialDistribution was not positive.')
        else:
            raise ValueError('The rate parameter given to an ' +\
                'ExponentialDistribution was not of a numerical type.')
        if type(shift) in numerical_types:
            self.shift = (1. * shift)
        else:
            raise ValueError('The shift given to an ' +\
                'ExponentialDistribution was not of numerical type.')
    
    @property
    def numparams(self):
        """
        Exponential distribution pdf is univariate so numparams always returns
        1.
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
        return\
            rand.exponential(scale=(1. / self.rate), size=shape) + self.shift
    
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
            return rate_close and shift_close
        else:
            return False
    
    def inverse_cdf(self, cdf):
        """
        Inverse of the cumulative distribution function.
        
        cdf: value between 0 and 1
        """
        return (self.shift - (np.log(1 - cdf) / self.rate))
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data about this distribution. The
        only things to save are the class name, rate, and shift.
        
        group: hdf5 file group to fill
        """
        group.attrs['class'] = 'ExponentialDistribution'
        group.attrs['rate'] = self.rate
        group.attrs['shift'] = self.shift

