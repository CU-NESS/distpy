"""
File: distpy/distribution/UniformDistribution.py
Author: Keith Tauscher
Date: 12 Feb 2018

Description: File containing a class representing a Weibull distribution.
"""
from __future__ import division
import numpy as np
import numpy.random as rand
from ..util import numerical_types
from .Distribution import Distribution

class WeibullDistribution(Distribution):
    """
    Class representing a Weibull distribution.
    """
    def __init__(self, shape=1, scale=1., metadata=None):
        """
        Creates a new WeibullDistribution with the given shape and scale.
        
        shape, scale: positive numbers. shape == 1 reduces to exponential
                      distribution
        """
        if type(shape) in numerical_types:
            if shape > 0:
                self.shape = shape
            else:
                raise ValueError("shape parameter of WeibullDistribution " +\
                    "was not positive.")
        else:
            raise TypeError("shape parameter of WeibullDistribution was " +\
                "not a number.")
        if type(scale) in numerical_types:
            if scale > 0:
                self.scale = scale
            else:
                raise ValueError("scale parameter of WeibullDistribution " +\
                    "was not positive.")
        else:
            raise TypeError("scale parameter of WeibullDistribution was " +\
                "not a number.")
        self.const_lp_term = np.log(self.shape) -\
            (self.shape * np.log(self.scale))
        self.metadata = metadata

    @property
    def numparams(self):
        """
        Weibull distribution is univariate so numparams always returns 1.
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
        return self.scale * random.weibull(self.shape, size=shape)


    def log_value(self, point):
        """
        Evaluates and returns the log of the value of this distribution when
        the variable is value.
        
        point: numerical value of the variable
        """
        if point >= 0:
            return self.const_lp_term + ((self.shape - 1) * np.log(point)) -\
                np.power(point / self.scale, self.shape)
        return -np.inf
    
    def to_string(self):
        """
        Finds and returns a string representation of this distribution.
        """
        return "Weibull({0:.2g}, {1:.2g})".format(self.shape, self.scale)
    
    def __eq__(self, other):
        """
        Checks for equality of this distribution with other. Returns True if
        other is a WeibullDistribution with the same shape and scale and False
         otherwise.
        """
        if isinstance(other, WeibullDistribution):
            tol_kwargs = {'rtol': 0., 'atol': 1e-9}
            shape_close = np.isclose(self.shape, other.shape, **tol_kwargs)
            scale_close = np.isclose(self.scale, other.scale, **tol_kwargs)
            metadata_equal = self.metadata_equal(other)
            return all([shape_close, scale_close, metadata_equal])
        else:
            return False
    
    def inverse_cdf(self, cdf):
        """
        Inverse of the cumulative distribution function.
        
        cdf: value between 0 and 1
        """
        return (self.scale * np.power(-np.log(1 - cdf), 1 / self.shape))
    
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
        Fills the given hdf5 file group with data from this distribution. All
        that needs to be saved is the class name and shape and scale values.
        
        group: hdf5 file group to fill
        save_metadata: if True, attempts to save metadata alongside
                                distribution and throws error if it fails
                       if False, metadata is ignored in saving process
        """
        group.attrs['class'] = 'WeibullDistribution'
        group.attrs['shape'] = self.shape
        group.attrs['scale'] = self.scale
        if save_metadata:
            self.save_metadata(group)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a WeibullDistribution from the given hdf5 file group.
        
        group: the same hdf5 file group which fill_hdf5_group was called on
               when this Distribution was saved
        
        returns: WeibullDistribution object created from the information in the
                 given group
        """
        try:
            assert group.attrs['class'] == 'WeibullDistribution'
        except:
            raise TypeError("The given hdf5 file doesn't seem to contain a " +\
                "WeibullDistribution.")
        metadata = Distribution.load_metadata(group)
        shape = group.attrs['shape']
        scale = group.attrs['scale']
        return WeibullDistribution(shape=shape, scale=scale, metadata=metadata)
    
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
        return (((self.shape - 1) / point) - ((self.shape / self.scale) *\
            ((point / self.scale) ** (self.shape - 1))))
    
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
        return (((1 - self.shape) / (point ** 2)) -\
            ((self.shape / self.scale) * ((self.shape - 1) / self.scale) *\
            ((point / self.scale) ** (self.shape - 2))))
    
    def copy(self):
        """
        Returns a deep copy of this Distribution. This function ignores
        metadata.
        """
        return WeibullDistribution(self.shape, self.scale)

