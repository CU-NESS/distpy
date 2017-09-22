"""
File: distpy/TruncatedGaussianDistribution.py
Author: Keith Tauscher
Date: 6 Aug 2017

Description: File containing class representing a truncated 1-dimensional
             gaussian distribution.
"""
import numpy as np
import numpy.random as rand
from scipy.special import erf, erfinv
from ..util import int_types
from .Distribution import Distribution

class TruncatedGaussianDistribution(Distribution):
    """
    Class representing a truncated Gaussian distribution. These are useful if
    one has some knowledge of the actual value of the parameter but also needs
    it to lie outside a given region.
    """
    def __init__(self, mean, var, low=None, high=None):
        """
        Initializes a new TruncatedGaussianDistribution using the given
        parameters.
        
        mean the mean of the (untruncated!) Gaussian
        variance the variance of the (untrucated!) Gaussian
        low lower limit of distribution. If None then it is assumed to be -inf
        high upper limit of distribution. If None then it is assumed to be +inf
        """
        self.mean = float(mean)
        self.var = float(var)
        
        if low is None:
            self.lo = None
            self._lo_term = -1.
        else:
            self.lo = float(low)
            self._lo_term = erf((self.lo - self.mean) / np.sqrt(2 * self.var))

        if high is None:
            self.hi = None
            self._hi_term = 1.
        else:
            self.hi = float(high)
            self._hi_term = erf((self.hi - self.mean) / np.sqrt(2 * self.var))

        self._cons_lp_term = -(np.log(np.pi * self.var / 2) / 2)
        self._cons_lp_term -= np.log(self._hi_term - self._lo_term)

    @property
    def numparams(self):
        """
        As of now, only univariate TruncatedGaussianDistribution's are
        implemented so numparams always returns 1.
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
        none_shape = (shape is None)
        if none_shape:
            shape = (1,)
        elif type(shape) in int_types:
            shape = (shape,)
        unifs = rand.rand(*shape)
        args_to_erfinv =\
            (unifs * self._hi_term) + ((1. - unifs) * self._lo_term)
        points = self.mean + (np.sqrt(2 * self.var) * erfinv(args_to_erfinv))
        if none_shape:
            return points[0]
        else:
            return points

    def log_value(self, point):
        """
        Evaluates and returns the log of the value of this distribution when
        the variable is point.
        
        point: numerical value of the variable
        """
        if (self.lo is not None and point < self.lo) or\
                (self.hi is not None and point > self.hi):
            return -np.inf
        return self._cons_lp_term - ((point - self.mean) ** 2) / (2 * self.var)

    def to_string(self):
        """
        Finds and returns string representation of this distribution.
        """
        if self.lo is None:
            low_string = "-inf"
        else:
            low_string = "{:.1g}".format(self.lo)
        if self.hi is None:
            high_string = "inf"
        else:
            high_string = "{:.1g}".format(self.hi)
        return "Normal({0:.2g}, {1:.2g}) on [{2!s},{3!s}]".format(self.mean,\
            self.var, low_string, high_string)
    
    def __eq__(self, other):
        """
        Checks for equality of this distribution to other. Returns True if
        other is a TruncatedGaussianDistribution with the same mean (down to
        10^-9 level) and variance (down to 10^-12 dynamic range), and hi and
        lo (down to 10^-9 level) and False otherwise.
        """
        if isinstance(other, TruncatedGaussianDistribution):
            mean_close = np.isclose(self.mean, other.mean, rtol=0, atol=1e-9)
            var_close = np.isclose(self.var, other.var, rtol=1e-12, atol=0)
            if self.hi is None:
                hi_close = (other.hi is None)
            elif other.hi is not None:
                hi_close = np.isclose(self.hi, other.hi, rtol=0, atol=1e-9)
            else:
                # since self.hi is not None in this block, just return False
                return False
            if self.lo is None:
                lo_close = (other.lo is None)
            elif other.lo is not None:
                lo_close = np.isclose(self.lo, other.lo, rtol=0, atol=1e-9)
            else:
                return False
            return mean_close and var_close and hi_close and lo_close
        else:
            return False
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data from this distribution. The
        low, high, mean, and variance values need to be saved along with the
        class name.
        """
        group.attrs['class'] = 'TruncatedGaussianDistribution'
        group.attrs['low'] = self.lo
        group.attrs['high'] = self.hi
        group.attrs['mean'] = self.mean
        group.attrs['variance'] = self.var

