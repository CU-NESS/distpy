"""
File: distpy/UniformDistribution.py
Author: Keith Tauscher
Date: 6 Aug 2017

Description: File containing a class representing a Weibull distribution.
"""
import numpy as np
import numpy.random as rand
from .TypeCategories import numerical_types
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
        return "Weibull(%.2g, %.2g)" % (self.shape, self.scale)
    
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
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data from this distribution. All
        that needs to be saved is the class name and shape and scale values.
        
        group: hdf5 file group to fill
        """
        group.attrs['class'] = 'WeibullDistribution'
        group.attrs['shape'] = self.shape
        group.attrs['scale'] = self.scale
