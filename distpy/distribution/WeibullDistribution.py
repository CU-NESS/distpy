"""
File: distpy/UniformDistribution.py
Author: Keith Tauscher
Date: 6 Aug 2017

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
    def __init__(self, shape=1, scale=1.):
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
        

    @property
    def numparams(self):
        """
        Weibull distribution is univariate so numparams always returns 1.
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
        return self.scale * rand.weibull(self.shape, size=shape)


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
            return np.allclose([self.shape, self.scale],\
                [other.shape, other.scale], rtol=0, atol=1e-9)
        else:
            return False
    
    def inverse_cdf(self, cdf):
        """
        Inverse of the cumulative distribution function.
        
        cdf: value between 0 and 1
        """
        return (self.scale * np.power(-np.log(1 - cdf), 1 / self.shape))
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data from this distribution. All
        that needs to be saved is the class name and shape and scale values.
        
        group: hdf5 file group to fill
        """
        group.attrs['class'] = 'WeibullDistribution'
        group.attrs['shape'] = self.shape
        group.attrs['scale'] = self.scale
    
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

