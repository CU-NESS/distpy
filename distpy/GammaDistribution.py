"""
File: distpy/GammaDistribution.py
Author: Keith Tauscher
Date: 6 Aug 2017

Description: File containing class representing a Gamma distribution.
"""
import numpy as np
import numpy.random as rand
from scipy.special import gammaln as log_gamma
from .TypeCategories import numerical_types
from .Distribution import Distribution

class GammaDistribution(Distribution):
    """
    A class representing a gamma distribution. This is useful for variables
    which are naturally non-negative.
    """
    def __init__(self, shape, scale=1.):
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

    @property
    def numparams(self):
        """
        Gamma distribution pdf is univariate, so numparams always returns 1.
        """
        return 1

    def draw(self):
        """
        Draws and returns a value from this distribution using numpy.random.
        """
        return rand.gamma(self.shape, scale=self.scale)

    def log_value(self, point):
        """
        Evaluates and returns the log of the value of this distribution when
        the variable is point.
        
        point: numerical value of the variable
        """
        return (self._shape_min_one * np.log(point)) -\
               (self.shape * np.log(self.scale)) -\
               (point / self.scale) - log_gamma(self.shape)
    
    def to_string(self):
        """
        Finds and returns the string representation of this GammaDistribution.
        """
        return "Gamma(%.2g, %.2g)" % (self.shape, self.scale)
    
    def __eq__(self, other):
        """
        Checks for equality between other and this object. Returns True if
        if other is a GammaDistribution with nearly the same shape and scale
        (up to dynamic range of 10^9) and False otherwise.
        """
        if isinstance(other, GammaDistribution):
            return np.allclose([self.shape, self.scale],\
                [other.shape, other.scale], rtol=1e-9, atol=0)
        else:
            return False
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data from this distribution. Only
        things to save are shape, scale, and class name.
        
        group: hdf5 file group to fill
        """
        group.attrs['class'] = 'GammaDistribution'
        group.attrs['shape'] = self.shape
        group.attrs['scale'] = self.scale

    def _check_if_greater_than_zero(self, value, name):
        #
        # Function which checks if the given value is positive.
        # If so, the function runs smoothly and returns nothing.
        # Otherwise, useful errors are raised.
        #
        if type(value) in numerical_types:
            if value <= 0:
                raise ValueError(("The %s given to " % (name,)) +\
                                 "a GammaDistribution wasn't positive.")
        else:
            raise ValueError(("The %s given to a " % (name,)) +\
                             "GammaDistribution wasn't of a numerical type.")

