"""
File: distpy/distribution/GammaDistribution.py
Author: Keith Tauscher
Date: Oct 15 2019

Description: File containing class representing a generalized Gamma
             distribution. The pdf has the form
             [(p/a)/Gamma(d/p)]*[(x/a)^(d-1)]*exp[-(x/a)^p] where d is shape,
             a is scale, and p is power.
"""
from __future__ import division
import numpy as np
import numpy.random as rand
from scipy.special import gamma, gammaln, gammaincinv
from ..util import numerical_types
from .Distribution import Distribution

class GammaDistribution(Distribution):
    """
    A class representing a generalized gamma distribution. This is useful for
    variables which are naturally non-negative.
    """
    def __init__(self, shape, scale=1, power=1, metadata=None):
        """
        Initializes a new gamma distribution with the given parameters.
        
        shape the exponent of x in the gamma pdf (must be greater than 0).
        scale amount to scale x by (x is divided by scale where it appears)
              (must be greater than 0).
        power power of point/scale in the exponent in the pdf
        metadata data to store with this distribution, should be hdf5-able
        """
        self.shape = shape
        self.scale = scale
        self.power = power
        self.metadata = metadata
    
    @staticmethod
    def create_from_mean_and_variance(mean, variance, metadata=None):
        """
        Creates a new GammaDistribution from the mean and variance (assuming
        power=1) instead of through the conventional shape and scale
        parameters.
        
        mean: positive number equal to mean of desired distribution
        variance: positive number equal to variance of desired distribution
        metadata: data to store with this distribution, should be hdf5-able
        
        returns: new GammaDistribution object encoding the desired distribution
        """
        shape = ((mean ** 2) / variance)
        scale = (variance / mean)
        return\
            GammaDistribution(shape, scale=scale, power=1, metadata=metadata)
    
    @property
    def shape(self):
        """
        Property storing the shape parameter of the gamma distribution. This is
        the power of x in the product in the pdf (not the power of x in the
        exponent of the pdf)
        """
        if not hasattr(self, '_shape'):
            raise AttributeError("shape was referenced before it was set.")
        return self._shape
    
    @property
    def shape_minus_one(self):
        """
        Property storing the shape parameter minus 1.
        """
        if not hasattr(self, '_shape_minus_one'):
            self._shape_minus_one = self.shape - 1
        return self._shape_minus_one
    
    @shape.setter
    def shape(self, value):
        """
        Setter for the shape parameter of the GammaDistribution.
        
        value: a positive number
        """
        if type(value) in numerical_types:
            if value > 0:
                self._shape = value
            else:
                raise ValueError("shape given to GammaDistribution was not " +\
                    "positive.")
        else:
            raise TypeError("shape given to GammaDistribution was not a " +\
                "number.")
    
    @property
    def scale(self):
        """
        Property storing the scale parameter of the gamma distribution. This is
        what x is divided by when it appears in the pdf.
        """
        if not hasattr(self, '_scale'):
            raise AttributeError("scale was referenced before it was set.")
        return self._scale
    
    @scale.setter
    def scale(self, value):
        """
        Setter for the scale parameter of the GammaDistribution.
        
        value: a positive number
        """
        if type(value) in numerical_types:
            if value > 0:
                self._scale = value
            else:
                raise ValueError("scale given to GammaDistribution was not " +\
                    "positive.")
        else:
            raise TypeError("scale given to GammaDistribution was not a " +\
                "number.")
    
    @property
    def power(self):
        """
        Property storing the power parameter of the gamma distribution. This is
        the power of x in the exponent in the pdf
        """
        if not hasattr(self, '_power'):
            raise AttributeError("power was referenced before it was set.")
        return self._power
    
    @power.setter
    def power(self, value):
        """
        Setter for the power parameter of the GammaDistribution.
        
        value: a positive number
        """
        if type(value) in numerical_types:
            if value > 0:
                self._power = value
            else:
                raise ValueError("power given to GammaDistribution was not " +\
                    "positive.")
        else:
            raise TypeError("power given to GammaDistribution was not a " +\
                "number.")
    
    @property
    def numparams(self):
        """
        Gamma distribution pdf is univariate, so numparams always returns 1.
        """
        return 1
    
    @property
    def mean(self):
        """
        Property storing the mean of this distribution.
        """
        if not hasattr(self, '_mean'):
            self._mean = self.scale *\
                np.exp(gammaln((self.shape + 1) / self.power) -\
                gammaln(self.shape / self.power))
        return self._mean
    
    @property
    def variance(self):
        """
        Property storing the covariance of this distribution.
        """
        if not hasattr(self, '_variance'):
            first_term = np.exp(gammaln((self.shape + 2) / self.power) -\
                gammaln(self.shape / self.power))
            second_term = (np.exp(gammaln((self.shape + 1) / self.power) -\
                gammaln(self.shape / self.power)) ** 2)
            self._variance = (self.scale ** 2) * (first_term - second_term)
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
        return self.inverse_cdf(random.uniform(size=shape))
    
    @property
    def log_value_constant(self):
        """
        Property storing the constant in the log pdf of this distribution.
        """
        if not hasattr(self, '_log_value_constant'):
            self._log_value_constant = np.log(self.power) -\
                (self.shape * np.log(self.scale)) -\
                gammaln(self.shape / self.power)
        return self._log_value_constant
    
    def log_value(self, point):
        """
        Evaluates and returns the log of the value of this distribution when
        the variable is point.
        
        point: numerical value of the variable
        """
        return self.log_value_constant +\
            (self.shape_minus_one * np.log(point)) -\
            np.power(point / self.scale, self.power)
    
    def to_string(self):
        """
        Finds and returns the string representation of this GammaDistribution.
        """
        return "Gamma({0:.2g}, {1:.2g}, {2:.2g})".format(self.shape,\
            self.scale, self.power)
    
    def __eq__(self, other):
        """
        Checks for equality between other and this object. Returns True if
        if other is a GammaDistribution with nearly the same shape, scale, and
        power (up to dynamic range of 10^9) and False otherwise.
        """
        if isinstance(other, GammaDistribution):
            tol_kwargs = {'rtol': 1e-9, 'atol': 0}
            shape_equal = np.isclose(self.shape, other.shape, **tol_kwargs)
            scale_equal = np.isclose(self.scale, other.scale, **tol_kwargs)
            power_equal = np.isclose(self.power, other.power, **tol_kwargs)
            metadata_equal = self.metadata_equal(other)
            return all([shape_equal, scale_equal, power_equal, metadata_equal])
        else:
            return False
    
    def inverse_cdf(self, cdf):
        """
        Inverse of the cumulative distribution function.
        
        cdf: value between 0 and 1
        """
        return self.scale *\
            np.power(gammaincinv(self.shape / self.power, cdf), 1 / self.power)
    
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
        Fills the given hdf5 file group with data from this distribution. Only
        things to save are shape, scale, and class name.
        
        group: hdf5 file group to fill
        save_metadata: if True, attempts to save metadata alongside
                                distribution and throws error if it fails
                       if False, metadata is ignored in saving process
        """
        group.attrs['class'] = 'GammaDistribution'
        group.attrs['shape'] = self.shape
        group.attrs['scale'] = self.scale
        group.attrs['power'] = self.power
        if save_metadata:
            self.save_metadata(group)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a GammaDistribution from the given hdf5 file group.
        
        group: the same hdf5 file group which fill_hdf5_group was called on
               when this Distribution was saved
        
        returns: a GammaDistribution object created from the information in the
                 given group
        """
        try:
            assert group.attrs['class'] == 'GammaDistribution'
        except:
            raise TypeError("The given hdf5 file doesn't seem to contain a " +\
                "GammaDistribution.")
        metadata = Distribution.load_metadata(group)
        shape = group.attrs['shape']
        scale = group.attrs['scale']
        power = group.attrs['power']
        return GammaDistribution(shape, scale=scale, power=power,\
            metadata=metadata)
    
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
        return ((self.shape_minus_one / point) -\
            ((self.power / self.scale) *\
            np.power(point / self.scale, self.power - 1)))
    
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
        return ((-1) * self.shape_minus_one) / np.power(point, 2) -\
            (((self.power * (self.power - 1)) / (self.scale ** 2)) *\
            np.power(point / self.scale, self.power - 2))
    
    def copy(self):
        """
        Returns a deep copy of this Distribution. This function ignores
        metadata.
        """
        return GammaDistribution(self.shape, self.scale, self.power)

