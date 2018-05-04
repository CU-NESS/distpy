"""
File: distpy/distribution/GammaDistribution.py
Author: Keith Tauscher
Date: 12 Feb 2018

Description: File containing class representing a Gamma distribution.
"""
import numpy as np
import numpy.random as rand
from scipy.special import gammaln, gammaincinv
from ..util import numerical_types
from .Distribution import Distribution

class GammaDistribution(Distribution):
    """
    A class representing a gamma distribution. This is useful for variables
    which are naturally non-negative.
    """
    def __init__(self, shape, scale=1., metadata=None):
        """
        Initializes a new gamma distribution with the given parameters.
        
        shape the exponent of x in the gamma pdf (must be greater than 0).
        scale amount to scale x by (x is divided by scale where it appears)
              (must be greater than 0).
        """
        self._check_if_greater_than_zero(shape, 'shape')
        self._check_if_greater_than_zero(scale, 'scale')
        self.shape = (shape * 1.)
        self._shape_min_one = self.shape - 1.
        self.scale = (scale * 1.)
        self.metadata = metadata

    @property
    def numparams(self):
        """
        Gamma distribution pdf is univariate, so numparams always returns 1.
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
        return random.gamma(self.shape, scale=self.scale, size=shape)

    def log_value(self, point):
        """
        Evaluates and returns the log of the value of this distribution when
        the variable is point.
        
        point: numerical value of the variable
        """
        return (self._shape_min_one * np.log(point)) -\
               (self.shape * np.log(self.scale)) -\
               (point / self.scale) - gammaln(self.shape)
    
    def to_string(self):
        """
        Finds and returns the string representation of this GammaDistribution.
        """
        return "Gamma({0:.2g}, {1:.2g})".format(self.shape, self.scale)
    
    def __eq__(self, other):
        """
        Checks for equality between other and this object. Returns True if
        if other is a GammaDistribution with nearly the same shape and scale
        (up to dynamic range of 10^9) and False otherwise.
        """
        if isinstance(other, GammaDistribution):
            tol_kwargs = {'rtol': 1e-9, 'atol': 0}
            shape_equal = np.isclose(self.shape, other.shape, **tol_kwargs)
            scale_equal = np.isclose(self.scale, other.scale, **tol_kwargs)
            metadata_equal = self.metadata_equal(other)
            return all([shape_equal, scale_equal, metadata_equal])
        else:
            return False
    
    def inverse_cdf(self, cdf):
        """
        Inverse of the cumulative distribution function.
        
        cdf: value between 0 and 1
        """
        return self.scale * gammaincinv(self.shape, cdf)
    
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
        return GammaDistribution(shape, scale=scale, metadata=metadata)

    def _check_if_greater_than_zero(self, value, name):
        #
        # Function which checks if the given value is positive.
        # If so, the function runs smoothly and returns nothing.
        # Otherwise, useful errors are raised.
        #
        if type(value) in numerical_types:
            if value <= 0:
                raise ValueError(("The {!s} given to a GammaDistribution " +\
                    "wasn't positive.").format(name))
        else:
            raise ValueError(("The {!s} given to a GammaDistribution " +\
                "wasn't of a numerical type.").format(name))
    
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
        return (((self.shape - 1.) / point) - (1. / self.scale))
    
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
        return ((1. - self.shape) / (point ** 2))

