"""
File: distpy/jumping/GaussianJumpingDistribution.py
Author: Keith Tauscher
Date: 12 Feb 2018

Description: File containing a jumping distribution which is Gaussian centered
             on the source point with a given covariance.
"""
import numpy as np
import numpy.linalg as la
from scipy.special import erf, erfinv
from ..util import create_hdf5_dataset, numerical_types, sequence_types
from .JumpingDistribution import JumpingDistribution

class TruncatedGaussianJumpingDistribution(JumpingDistribution):
    """
    Class representing a jumping distribution which is centered on the source
    point and has the given covariance.
    """
    def __init__(self, variance, low=None, high=None):
        """
        Initializes a TruncatedGaussianJumpingDistribution with the given
        variance.
        
        variance: a single number representing the variance of the
                  non-truncated Gaussian
        """
        self.variance = variance
        self.low = low
        self.high = high
    
    @property
    def low(self):
        """
        Property storing the low endpoint of this truncated Gaussian. If none
        exists, it is -np.inf
        """
        if not hasattr(self, '_low'):
            raise AttributeError("low referenced before it was set.")
        return self._low
    
    @low.setter
    def low(self, value):
        """
        Setter for the low endpoint of this truncated Gaussian.
        
        value: either None (if there is no low endpoint) or a real number
        """
        if value is None:
            self._low = -np.inf
        elif type(value) in numerical_types:
            self._low = value
        else:
            raise TypeError("low was neither None nor a number.")
    
    @property
    def high(self):
        """
        Property storing the high endpoint of this truncated Gaussian. If none
        exists, it is +np.inf
        """
        if not hasattr(self, '_high'):
            raise AttributeError("high referenced before it was set.")
        return self._high
    
    @high.setter
    def high(self, value):
        """
        Setter for the high endpoint of this truncated Gaussian.
        
        value: either None (if there is no high endpoint) or a real number
               larger than low
        """
        if value is None:
            self._high = np.inf
        elif type(value) in numerical_types:
            if value > self.low:
                self._high = value
            else:
                raise ValueError("high was not larger than low.")
        else:
            raise TypeError("high was neither None nor a number.")
    
    @property
    def variance(self):
        """
        Property storing the variance of the non-truncated Gaussian.
        """
        if not hasattr(self, '_variance'):
            raise AttributeError("variance referenced before it was set.")
        return self._variance
    
    @variance.setter
    def variance(self, value):
        """
        Sets the variance of this TruncatedGaussianJumpingDistribution
        
        value: either a single number (if this GaussianJumpingDistribution
               should be 1D) or a square 2D array
        """
        if type(value) in numerical_types:
            self._variance = value
        else:
            raise TypeError("variance was not a number.")
    
    @property
    def root_twice_variance(self):
        """
        """
        if not hasattr(self, '_root_twice_variance'):
            self._root_twice_variance = np.sqrt(2 * self.variance)
        return self._root_twice_variance
    
    def left_erf(self, source):
        """
        Computes erf(low-source/root_twice_variance)
        
        source: the mean of the truncated Gaussian
        
        returns: a single number
        """
        if self.low == -np.inf:
            return (-1.)
        else:
            return erf((self.low - source) / self.root_twice_variance)
    
    def right_erf(self, source):
        """
        Computes erf(high-source/root_twice_variance)
        
        source: the mean of the truncated Gaussian
        
        returns: a single number
        """
        if self.high == np.inf:
            return 1.
        else:
            return erf((self.high - source) / self.root_twice_variance)
    
    def erf_difference(self, source):
        """
        Computes right_erf(source)-left_erf(source)
        
        source: the mean of the truncated Gaussian
        
        returns: a single number between 0 (exclusive) and 2 (inclusive)
        """
        return (self.right_erf(source) - self.left_erf(source))
    
    @property
    def constant_in_log_value(self):
        """
        """
        if not hasattr(self, '_constant_in_log_value'):
            self._constant_in_log_value =\
                (np.log(2. / (np.pi * self.variance)) / 2.)
        return self._constant_in_log_value
    
    def draw(self, source, shape=None):
        """
        Draws a destination point from this jumping distribution given a source
        point.
        
        source: if this JumpingDistribution is univariate, source should be a
                                                           single number
                otherwise, source should be numpy.ndarray of shape (numparams,)
        shape: if None, returns single random variate
                        (scalar for univariate ; 1D array for multivariate)
               if int, n, returns n random variates
                          (1D array for univariate ; 2D array for multivariate)
               if tuple of n ints, returns that many random variates
                                   n-D array for univariate ;
                                   (n+1)-D array for multivariate
        
        returns: either single value (if distribution is 1D) or array of values
        """
        uniforms = np.random.uniform(size=shape)
        erfinv_argument = ((uniforms * self.right_erf(source)) +\
            ((1 - uniforms) * self.left_erf(source)))
        return (source + (self.root_twice_variance * erfinv(erfinv_argument)))
    
    def log_value(self, source, destination):
        """
        Computes the log-PDF: ln(f(source->destination))
        
        source, destination: either single values (if distribution is 1D) or
                             arrays of values
        
        returns: single number, logarithm of value of this distribution at the
                 given point
        """
        difference = (destination - source)
        return (self.constant_in_log_value +\
            (((difference / self.standard_deviation) ** 2) / (-2.))) -\
            np.log(self.erf_difference(source))
    
    def log_value_difference(self, source, destination):
        """
        Computes the log-PDF difference:
        ln(f(source->destination)/f(destination->source))
        
        source, destination: either single values (if distribution is 1D) or
                             arrays of values
        
        returns: single number difference between one-way log-PDF's
        """
        return np.log(\
            self.erf_difference(destination) / self.erf_difference(source))
    
    @property
    def numparams(self):
        """
        Property storing the integer number of parameters described by this
        distribution. Since the truncated Gaussian is only easily analytically
        sampled in the case of 1 parameter, these Truncated Gaussians only
        allow 1 parameter.
        """
        return 1
    
    @property
    def standard_deviation(self):
        """
        Property storing the square root of the variance (in the case that
        numparams == 1). If this Gaussian is multivariate, referencing this
        property will throw a NotImplementedError because the standard
        deviation is not well defined in this case.
        """
        if not hasattr(self, '_standard_deviation'):
            self._standard_deviation = np.sqrt(self.variance)
        return self._standard_deviation
    
    def __eq__(self, other):
        """
        Tests for equality between this distribution and other. All subclasses
        must implement this function.
        
        other: JumpingDistribution with which to check for equality
        
        returns: True or False
        """
        if isinstance(other, TruncatedGaussianJumpingDistribution):
            if self.numparams == other.numparams:
                variances_close = np.allclose(self.variance, other.variance,\
                    rtol=1e-12, atol=1e-12)
                lows_close = np.isclose(self.low, other.low, atol=1e-6)
                highs_close = np.isclose(self.high, other.high, atol=1e-6)
                return (variances_close and lows_close and highs_close)
            else:
                return False
        else:
            return False
    
    @property
    def is_discrete(self):
        """
        Property storing boolean describing whether this JumpingDistribution
        describes discrete (True) or continuous (False) variable(s).
        """
        return False
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data from this distribution. The
        fact that this is a GaussianJumpingDistribution is saved along with the
        mean and covariance.
        
        group: hdf5 file group to fill
        """
        group.attrs['class'] = 'TruncatedGaussianJumpingDistribution'
        group.attrs['variance'] = self.variance
        if self.low != -np.inf:
            group.attrs['low'] = self.low
        if self.high != np.inf:
            group.attrs['high'] = self.high
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a TruncatedGaussianJumpingDistribution from the given hdf5 file
        group.
        
        group: the same hdf5 file group which fill_hdf5_group was called on
               when this TruncatedGaussianJumpingDistribution was saved
        
        returns: a TruncatedGaussianJumpingDistribution object created from the
                 information in the given group
        """
        try:
            assert\
                group.attrs['class'] == 'TruncatedGaussianJumpingDistribution'
        except:
            raise ValueError("The given group does not seem to contain a " +\
                "TruncatedGaussianJumpingDistribution.")
        variance = group.attrs['variance']
        if 'low' in group.attrs:
            low = group.attrs['low']
        else:
            low = None
        if 'high' in group.attrs:
            high = group.attrs['high']
        else:
            high = None
        return\
            TruncatedGaussianJumpingDistribution(variance, low=low, high=high)

