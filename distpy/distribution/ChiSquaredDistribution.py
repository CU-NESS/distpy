"""
File: distpy/ChiSquaredDistribution.py
Author: Keith Tauscher
Date: 6 Aug 2017

Description: File containing class representing a chi-squared distribution.
"""
import numpy as np
import numpy.random as rand
from scipy.special import gammaln as log_gamma
from ..util import int_types
from .Distribution import Distribution

class ChiSquaredDistribution(Distribution):
    """
    Class representing a chi-squared distribution. This is useful for variables
    variables which are naturally positive or are representable as the sum of
    squared gaussian-distributed variables. Its only parameter is the number of
    degrees of freedom, a positive integer.
    """
    def __init__(self, degrees_of_freedom):
        """
        Initializes a new chi-squared distribution with the given parameters.
        
        degrees_of_freedom: positive integer
        """
        if type(degrees_of_freedom) in int_types:
            if degrees_of_freedom > 0:
                self.degrees_of_freedom = degrees_of_freedom
            else:
                raise ValueError("degrees_of_freedom_given to " +\
                    "ChiSquaredDistribution was not positive.")
        else:
            raise ValueError("degrees_of_freedom given to " +\
                "ChiSquaredDistribution was not an integer.")
        self.const_lp_term = -(self.degrees_of_freedom * (np.log(2) / 2)) -\
            log_gamma(self.degrees_of_freedom / 2.)
    
    @property
    def numparams(self):
        """
        Chi-squared pdf is univariate, so numparams always returns 1.
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
        return rand.chisquare(self.degrees_of_freedom, size=shape)

    def log_value(self, point):
        """
        Evaluates and returns the log of the value of this distribution when
        the variable is point.
        
        point: numerical value of the variable
        """
        return self.const_lp_term - (point / 2) +\
            (((self.degrees_of_freedom / 2.) - 1) * np.log(point))
    
    def to_string(self):
        """
        Finds and returns the string representation of this
        ChiSquaredDistribution.
        """
        return "ChiSquared({})".format(self.degrees_of_freedom)
    
    def __eq__(self, other):
        """
        Checks for equality between other and this object. Returns True if
        if other is a ChiSquaredDistribution with the same degrees_of_freedom.
        """
        if isinstance(other, ChiSquaredDistribution):
            return self.degrees_of_freedom == other.degrees_of_freedom
        else:
            return False
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data from this distribution. Only
        thing to save is the number of degrees of freedom.
        
        group: hdf5 file group to fill
        """
        group.attrs['class'] = 'ChiSquaredDistribution'
        group.attrs['degrees_of_freedom'] = self.degrees_of_freedom

