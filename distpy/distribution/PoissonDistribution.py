"""
File: distpy/PoissonDistribution.py
Author: Keith Tauscher
Date: 6 Aug 2017

Description: File containing class representing a Poisson distribution.
"""
import numpy as np
import numpy.random as rand
from scipy.special import gammaln as log_gamma
from ..util import int_types, numerical_types
from .Distribution import Distribution

class PoissonDistribution(Distribution):
    """
    Distribution with support on the nonnegative integers. It has only one
    parameter, the scale, which is both its mean and its variance.
    """
    def __init__(self, scale):
        """
        Initializes new PoissonDistribution with given scale.
        
        scale: mean and variance of distribution (must be positive)
        """
        if type(scale) in numerical_types:
            if scale > 0:
                self.scale = (scale * 1.)
            else:
                raise ValueError("scale given to PoissonDistribution was " +\
                    "not positive.")
        else:
            raise ValueError("scale given to PoissonDistribution was not a " +\
                "number.")
    
    @property
    def numparams(self):
        """
        Poisson distribution pdf is univariate so numparams always returns 1.
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
        return rand.poisson(lam=self.scale, size=shape)
    
    def log_value(self, point):
        """
        Evaluates and returns the log of the value of this distribution when
        the variable is point.
        
        point: numerical value of the variable
        """
        if type(point) in int_types:
            if point >= 0:
                return (point * np.log(self.scale)) - self.scale -\
                    log_gamma(point + 1)
            else:
                return -np.inf
        else:
            raise TypeError("point given to PoissonDistribution was not an " +\
                "integer.")

    def to_string(self):
        """
        Finds and returns a string version of this PoissonDistribution.
        """
        return "Poisson({:.4g})".format(self.scale)
    
    def __eq__(self, other):
        """
        Checks for equality of this distribution with other. Returns True if
        other is a PoissonDistribution with the same scale.
        """
        if isinstance(other, PoissonDistribution):
            return np.isclose(self.scale, other.scale, rtol=1e-6, atol=1e-6)
        else:
            return False
    
    @property
    def can_give_confidence_intervals(self):
        """
        In distpy, confidence intervals are not supported with discrete
        distributions.
        """
        return False
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data about this distribution. The
        only thing to save is the scale.
        
        group: hdf5 file group to fill
        """
        group.attrs['class'] = 'PoissonDistribution'
        group.attrs['scale'] = self.scale
    
    @property
    def gradient_computable(self):
        """
        Property which stores whether the gradient of the given distribution
        has been implemented. Since this is a discrete distribution, it returns
        False.
        """
        return False 
    
    @property
    def hessian_computable(self):
        """
        Property which stores whether the hessian of the given distribution
        has been implemented. Since this is a discrete distribution, it returns
        False.
        """
        return False

