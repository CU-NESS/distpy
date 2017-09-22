"""
File: distpy/GeometricDistribution.py
Author: Keith Tauscher
Date: 6 Aug 2017

Description: File containing class representing a geometric distribution.
"""
import numpy as np
import numpy.random as rand
from ..util import int_types, numerical_types
from .Distribution import Distribution

class GeometricDistribution(Distribution):
    """
    Distribution with support on the non-negative integers. It has only one
    parameter, the common ratio between successive probabilities.
    """
    def __init__(self, common_ratio):
        """
        Initializes new GeometricDistribution with given scale.
        
        common_ratio: ratio between successive probabilities
        """
        if type(common_ratio) in numerical_types:
            if (common_ratio > 0.) and (common_ratio < 1.):
                self.common_ratio = common_ratio
            else:
                raise ValueError("scale given to GeometricDistribution was " +\
                    "not between 0 and 1.")
        else:
            raise ValueError("common_ratio given to GeometricDistribution " +\
                "was not a number.")
        self.const_lp_term = np.log(1 - self.common_ratio)
    
    @property
    def numparams(self):
        """
        Geometric distribution pdf is univariate so numparams always returns 1.
        """
        return 1
    
    @property
    def log_common_ratio(self):
        """
        Property storing the natural logarithm of the common ratio of
        successive probabilities.
        """
        if not hasattr(self, '_log_common_ratio'):
            self._log_common_ratio = np.log(self.common_ratio)
        return self._log_common_ratio
    
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
        return rand.geometric(1 - self.common_ratio, size=shape) - 1
    
    def log_value(self, point):
        """
        Evaluates and returns the log of the value of this distribution when
        the variable is point.
        
        point: numerical value of the variable
        """
        if type(point) in int_types:
            if point >= 0:
                return self.const_lp_term + (point * self.log_common_ratio)
            else:
                return -np.inf
        else:
            raise TypeError("point given to GeometricDistribution was not " +\
                "an integer.")

    def to_string(self):
        """
        Finds and returns a string version of this GeometricDistribution.
        """
        return "Geometric({:.4g})".format(self.common_ratio)
    
    def __eq__(self, other):
        """
        Checks for equality of this distribution with other. Returns True if
        other is a GeometricDistribution with the same scale.
        """
        if isinstance(other, GeometricDistribution):
            return np.isclose(self.common_ratio, other.common_ratio, rtol=0,\
                atol=1e-6)
        else:
            return False
    
    @property
    def can_give_confidence_intervals(self):
        """
        In distpy, discrete distributions do not support confidence intervals.
        """
        return False
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data about this distribution. The
        only thing to save is the common_ratio.
        
        group: hdf5 file group to fill
        """
        group.attrs['class'] = 'GeometricDistribution'
        group.attrs['common_ratio'] = self.common_ratio

