"""
File: distpy/distribution/SechSquaredDistribution.py
Author: Keith Tauscher
Date: 8 Jul 2018

Description: File containing class representing a sech squared distribution.
"""
import numpy as np
import numpy.random as rand
from ..util import numerical_types
from .Distribution import Distribution

class SechSquaredDistribution(Distribution):
    """
    A class representing a sech squared distribution, a tail-heavy symmetric
    distribution.
    """
    def __init__(self, mean, variance, metadata=None):
        """
        Initializes a new gamma distribution with the given parameters.
        
        mean: center of symmetric distribution
        variance: squared width of desired distribution
        """
        self.mean = mean
        self.variance = variance
        self.metadata = metadata
    
    @property
    def mean(self):
        """
        Property storing the mean of this distribution.
        """
        if not hasattr(self, '_mean'):
            raise AttributeError("mean was referenced before it was set.")
        return self._mean
    
    @mean.setter
    def mean(self, value):
        """
        Setter for the mean of this distribution.
        
        value: real number center of the distribution
        """
        if type(value) in numerical_types:
            self._mean = value
        else:
            raise TypeError("mean was set to a non-number.")
    
    @property
    def variance(self):
        """
        Property storing the variance of this distribution.
        """
        if not hasattr(self, '_variance'):
            raise AttributeError("variance was referenced before it was set.")
        return self._variance
    
    @variance.setter
    def variance(self, value):
        """
        Setter for the variance of this distribution.
        
        value: positive number squared width of the distribution
        """
        if type(value) in numerical_types:
            if value > 0:
                self._variance = value
            else:
                raise ValueError("variance was set to a non-positive number.")
        else:
            raise TypeError("variance was set to a non-number.")
    
    @property
    def scale(self):
        """
        Property storing a measure of the scale of the distribution.
        """
        if not hasattr(self, '_scale'):
            self._scale = np.sqrt(3 * self.variance) / np.pi
        return self._scale
    
    @property
    def log_value_constant(self):
        """
        Property storing the constant part of the log value of this
        distribution (this is related to the log of the normalization constant
        in the pdf).
        """
        if not hasattr(self, '_log_value_constant'):
            self._log_value_constant = ((-1) * np.log(2 * self.scale))
        return self._log_value_constant
    
    @property
    def numparams(self):
        """
        Sech squared distribution pdf is univariate, so numparams always
        returns 1.
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
        return self.mean +\
            (self.scale * np.arctanh((2 * random.uniform(size=shape)) - 1))

    def log_value(self, point):
        """
        Evaluates and returns the log of the value of this distribution when
        the variable is point.
        
        point: numerical value of the variable
        """
        return self.log_value_constant -\
            (2 * np.log(np.cosh((point - self.mean) / self.scale)))
    
    def to_string(self):
        """
        Finds and returns the string representation of this
        SechSquaredDistribution.
        """
        return "Sech2({0:.2g}, {1:.2g})".format(self.mean, self.variance)
    
    def __eq__(self, other):
        """
        Checks for equality between other and this object. Returns True if
        if other is a SechSquaredDistribution with nearly the same mean and
        variance (up to dynamic range of 10^9) and False otherwise.
        """
        if isinstance(other, SechSquaredDistribution):
            tol_kwargs = {'rtol': 1e-9, 'atol': 0}
            mean_equal = np.isclose(self.mean, other.mean, **tol_kwargs)
            variance_equal =\
                np.isclose(self.variance, other.variance, **tol_kwargs)
            metadata_equal = self.metadata_equal(other)
            return all([mean_equal, variance_equal, metadata_equal])
        else:
            return False
    
    def inverse_cdf(self, cdf):
        """
        Inverse of the cumulative distribution function.
        
        cdf: value between 0 and 1
        """
        return self.mean + (self.scale * np.arctanh((2 * cdf) - 1))
    
    @property
    def minimum(self):
        """
        Property storing the minimum allowable value(s) in this distribution.
        """
        return None
    
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
        Fills the given hdf5 file group with data from this distribution. Only
        things to save are shape, scale, and class name.
        
        group: hdf5 file group to fill
        save_metadata: if True, attempts to save metadata alongside
                                distribution and throws error if it fails
                       if False, metadata is ignored in saving process
        """
        group.attrs['class'] = 'SechSquaredDistribution'
        group.attrs['mean'] = self.mean
        group.attrs['variance'] = self.variance
        if save_metadata:
            self.save_metadata(group)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a SechSquaredDistribution from the given hdf5 file group.
        
        group: the same hdf5 file group which fill_hdf5_group was called on
               when this Distribution was saved
        
        returns: a SechSquaredDistribution object created from the information
                 in the given group
        """
        try:
            assert group.attrs['class'] == 'SechSquaredDistribution'
        except:
            raise TypeError("The given hdf5 file doesn't seem to contain a " +\
                "SechSquaredDistribution.")
        metadata = Distribution.load_metadata(group)
        mean = group.attrs['mean']
        variance = group.attrs['variance']
        return SechSquaredDistribution(mean, variance, metadata=metadata)
    
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
        return ((-2) / self.scale) * np.tanh((point - self.mean) / self.scale)
    
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
        return (((-2) * (np.pi ** 2)) / (3 * self.variance)) *\
            np.power((point - self.mean) / self.scale, -2)
    
    def copy(self):
        """
        Returns a deep copy of this Distribution. This function ignores
        metadata.
        """
        return SechSquaredDistribution(self.mean, self.variance)

