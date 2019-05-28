"""
File: distpy/distribution/TruncatedGaussianDistribution.py
Author: Keith Tauscher
Date: 12 Feb 2018

Description: File containing class representing a truncated 1-dimensional
             gaussian distribution.
"""
import numpy as np
import numpy.random as rand
from scipy.special import erf, erfinv
from ..util import int_types, numerical_types
from .Distribution import Distribution

class TruncatedGaussianDistribution(Distribution):
    """
    Class representing a truncated Gaussian distribution. These are useful if
    one has some knowledge of the actual value of the parameter but also needs
    it to lie outside a given region.
    """
    def __init__(self, mean, variance, low=None, high=None, metadata=None):
        """
        Initializes a new TruncatedGaussianDistribution using the given
        parameters.
        
        mean the mean of the (untruncated!) Gaussian
        variance the variance of the (untrucated!) Gaussian
        low lower limit of distribution. If None then it is assumed to be -inf
        high upper limit of distribution. If None then it is assumed to be +inf
        """
        self.mean = mean
        self.variance = variance
        self.low = low
        self.high = high
        self.metadata = metadata
    
    @property
    def low(self):
        """
        Property storing the lowest allowable value drawn from this
        distribution.
        """
        if not hasattr(self, '_low'):
            raise AttributeError("low was referenced before it was set.")
        return self._low
    
    @low.setter
    def low(self, value):
        """
        Setter for the lowest allowable value drawn from this distribution.
        
        value: a real number
        """
        if type(value) is type(None):
            self._low = None
        elif type(value) in numerical_types:
            self._low = value
        else:
            raise ValueError("low was set to neither None nor a number.")
    
    @property
    def high(self):
        """
        Property storing the highest allowable value drawn from this
        distribution.
        """
        if not hasattr(self, '_high'):
            raise AttributeError("high was referenced before it was set.")
        return self._high
    
    @high.setter
    def high(self, value):
        """
        Setter for the highest allowable value drawn from this distribution.
        
        value: a real number
        """
        if type(value) is type(None):
            self._high = None
        elif type(value) in numerical_types:
            if value > self.low:
                self._high = value
            else:
                raise ValueError("high was set to a number less than or " +\
                    "equal to low.")
        else:
            raise ValueError("high was set to neither None nor a number.")
    
    @property
    def const_lp_term(self):
        """
        Property storing the constant part of the log probability density of
        this distribution.
        """
        if not hasattr(self, '_const_lp_term'):
            self._const_lp_term =\
                ((-1) * (np.log(np.pi * self.variance / 2) / 2)) -\
                np.log(self.high_term - self.low_term)
        return self._const_lp_term
    
    @property
    def low_term(self):
        """
        Property storing the scaled error function at the low point.
        """
        if not hasattr(self, '_low_term'):
            if type(self.low) is type(None):
                self._low_term = -1
            else:
                self._low_term =\
                    erf((self.low - self.mean) / np.sqrt(2 * self.variance))
        return self._low_term
    
    @property
    def high_term(self):
        """
        Property storing the scaled error function at the high point.
        """
        if not hasattr(self, '_high_term'):
            if type(self.high) is type(None):
                self._high_term = 1
            else:
                self._high_term =\
                    erf((self.high - self.mean) / np.sqrt(2 * self.variance))
        return self._high_term
    
    @property
    def mean(self):
        """
        Property storing the mean of the untruncated Gaussian used.
        """
        if not hasattr(self, '_mean'):
            raise AttributeError("mean was referenced before it was set.")
        return self._mean
    
    @mean.setter
    def mean(self, value):
        """
        Setter for the mean of the untruncated Gaussian used.
        
        value: any real number
        """
        if type(value) in numerical_types:
            self._mean = (value * 1.)
        else:
            raise TypeError("mean was set to a non-number.")
    
    @property
    def variance(self):
        """
        Property storing the variance of the untruncated Gaussian used.
        """
        if not hasattr(self, '_variance'):
            raise AttributeError("variance was referenced before it was set.")
        return self._variance
    
    @variance.setter
    def variance(self, value):
        """
        Setter for the variance of the untruncated Gaussian used.
        
        value: any positive number
        """
        if type(value) in numerical_types:
            if value > 0:
                self._variance = (value * 1.)
            else:
                raise ValueError("variance must be positive.")
        else:
            raise TypeError("variance was set to a non-number.")

    @property
    def numparams(self):
        """
        As of now, only univariate TruncatedGaussianDistribution's are
        implemented so numparams always returns 1.
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
        none_shape = (type(shape) is type(None))
        if none_shape:
            shape = (1,)
        elif type(shape) in int_types:
            shape = (shape,)
        unifs = random.rand(*shape)
        args_to_erfinv =\
            (unifs * self.high_term) + ((1. - unifs) * self.low_term)
        points =\
            self.mean + (np.sqrt(2 * self.variance) * erfinv(args_to_erfinv))
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
        if (type(self.low) is not type(None) and point < self.low) or\
                (type(self.high) is not type(None) and point > self.high):
            return -np.inf
        return (self.const_lp_term -\
            ((point - self.mean) ** 2) / (2 * self.variance))

    def to_string(self):
        """
        Finds and returns string representation of this distribution.
        """
        if type(self.low) is type(None):
            low_string = "-inf"
        else:
            low_string = "{:.1g}".format(self.low)
        if type(self.high) is type(None):
            high_string = "inf"
        else:
            high_string = "{:.1g}".format(self.high)
        return "Normal({0:.2g}, {1:.2g}) on [{2!s},{3!s}]".format(self.mean,\
            self.variance, low_string, high_string)
    
    def __eq__(self, other):
        """
        Checks for equality of this distribution to other. Returns True if
        other is a TruncatedGaussianDistribution with the same mean (down to
        10^-9 level) and variance (down to 10^-12 dynamic range), and hi and
        lo (down to 10^-9 level) and False otherwise.
        """
        if isinstance(other, TruncatedGaussianDistribution):
            mean_close = np.isclose(self.mean, other.mean, rtol=0, atol=1e-9)
            variance_close =\
                np.isclose(self.variance, other.variance, rtol=1e-12, atol=0)
            if type(self.high) is type(None):
                hi_close = (type(other.high) is type(None))
            elif type(other.high) is not type(None):
                hi_close = np.isclose(self.high, other.high, rtol=0, atol=1e-9)
            else:
                # since self.high is not None in this block, just return False
                return False
            if type(self.low) is type(None):
                lo_close = (type(other.low) is type(None))
            elif type(other.low) is not type(None):
                lo_close = np.isclose(self.low, other.low, rtol=0, atol=1e-9)
            else:
                return False
            metadata_equal = self.metadata_equal(other)
            return all([mean_close, variance_close, hi_close, lo_close,\
                metadata_equal])
        else:
            return False
    
    def inverse_cdf(self, cdf):
        """
        Inverse of cumulative distribution function.
        
        cdf: value between 0 and 1
        """
        erfinv_args =\
            (self.low_term + (cdf * (self.high_term - self.low_term)))
        return (self.mean + (np.sqrt(2 * self.variance) * erfinv(erfinv_args)))
    
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
        Fills the given hdf5 file group with data from this distribution. The
        low, high, mean, and variance values need to be saved along with the
        class name.
        
        group: hdf5 file group to fill
        save_metadata: if True, attempts to save metadata alongside
                                distribution and throws error if it fails
                       if False, metadata is ignored in saving process
        """
        group.attrs['class'] = 'TruncatedGaussianDistribution'
        if type(self.low) is not type(None):
            group.attrs['low'] = self.low
        if type(self.high) is not type(None):
            group.attrs['high'] = self.high
        group.attrs['mean'] = self.mean
        group.attrs['variance'] = self.variance
        if save_metadata:
            self.save_metadata(group)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a TruncatedGaussianDistribution from the given hdf5 file group.
        
        group: the same hdf5 file group which fill_hdf5_group was called on
               when this Distribution was saved
        
        returns: a TruncatedGaussianDistribution object created from the
                 information in the given group
        """
        try:
            assert group.attrs['class'] == 'TruncatedGaussianDistribution'
        except:
            raise TypeError("The given hdf5 file doesn't seem to contain a " +\
                "TruncatedGaussianDistribution.")
        metadata = Distribution.load_metadata(group)
        mean = group.attrs['mean']
        variance = group.attrs['variance']
        if 'low' in group.attrs:
            low = group.attrs['low']
        else:
            low = None
        if 'high' in group.attrs:
            high = group.attrs['high']
        else:
            high = None
        return TruncatedGaussianDistribution(mean, variance, low=low,\
            high=high, metadata=metadata)
    
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
        return (self.mean - point) / self.variance
    
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
        return (-1.) / self.variance
    
    def copy(self):
        """
        Returns a deep copy of this Distribution. This function ignores
        metadata.
        """
        return TruncatedGaussianDistribution(self.mean, self.variance,\
            self.low, self.hi)

