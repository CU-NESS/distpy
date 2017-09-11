"""
File: distpy/BetaDistribution.py
Author: Keith Tauscher
Date: 6 Aug 2017

Description: File containing class representing a beta distribution.
"""
import numpy as np
import numpy.random as rand
from scipy.special import beta as beta_func
from ..util import numerical_types
from .Distribution import Distribution

class BetaDistribution(Distribution):
    """
    Class representing a beta distribution. Useful for parameters which must
    lie between 0 and 1. Classically, this is the distribution of the
    probability of success in a binary experiment where alpha successes and
    beta failures have been observed.
    """
    def __init__(self, alpha, beta):
        """
        Initializes a new BetaDistribution.
        
        alpha, beta parameters representing number of successes/failures
                    (both must be greater than 0)
        """
        if (type(alpha) in numerical_types) and\
            (type(beta) in numerical_types):
            if (alpha >= 0) and (beta >= 0):
                self.alpha = (alpha * 1.)
                self.beta  = (beta * 1.)
                self._alpha_min_one = self.alpha - 1.
                self._beta_min_one = self.beta - 1.
            else:
                raise ValueError('The alpha or beta parameter given ' +\
                                 'to a Beta was not non-negative.')
        else:
            raise ValueError('The alpha or beta parameter given to a ' +\
                             'Beta were not of a numerical type.')
        self.const_lp_term = -np.log(beta_func(self.alpha, self.beta))
    
    @property
    def numparams(self):
        """
        Beta distribution pdf is univariate, so numparams always returns 1.
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
        return rand.beta(self.alpha, self.beta, size=shape)
    
    def log_value(self, point):
        """
        Evaluates and returns the log of the value of this distribution when
        the variable is point.
        
        point: numerical value of the variable
        """
        if (point <= 0) or (point >= 1):
            return -np.inf
        return self.const_lp_term + (self._alpha_min_one * np.log(point)) +\
               (self._beta_min_one * np.log(1. - point))
    
    def to_string(self):
        """
        Finds and returns a string representation of this BetaDistribution.
        """
        return "Beta(%.2g, %.2g)" % (self.alpha, self.beta)
    
    def __eq__(self, other):
        """
        Checks for equality of this object with other. Returns True if other is
        a BetaDistribution with nearly the same alpha and beta (down to 10^-9
        level) and False otherwise.
        """
        if isinstance(other, BetaDistribution):
            return np.isclose([self.alpha, self.beta],\
                [other.alpha, other.beta], rtol=0, atol=1e-9)
        else:
            return False
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 group with data from this distribution. All that
        is to be saved is the class name, alpha, and beta.
        
        group: hdf5 file group to fill
        """
        group.attrs['class'] = 'BetaDistribution'
        group.attrs['alpha'] = self.alpha
        group.attrs['beta'] = self.beta

