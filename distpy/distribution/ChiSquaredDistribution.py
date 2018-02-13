"""
File: distpy/distribution/ChiSquaredDistribution.py
Author: Keith Tauscher
Date: Feb 12 2017

Description: File containing class representing a chi-squared distribution.
"""
from __future__ import division
import numpy as np
import numpy.random as rand
from scipy.special import gammaln, gammaincinv
from ..util import bool_types, int_types
from .Distribution import Distribution

class ChiSquaredDistribution(Distribution):
    """
    Class representing a chi-squared distribution. This is useful for variables
    variables which are naturally positive or are representable as the sum of
    squared gaussian-distributed variables. Its only parameter is the number of
    degrees of freedom, a positive integer.
    """
    def __init__(self, degrees_of_freedom, reduced=False, metadata=None):
        """
        Initializes a new chi-squared distribution with the given parameters.
        
        degrees_of_freedom: positive integer
        """
        self.reduced = reduced
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
            gammaln(self.degrees_of_freedom / 2.)
        if self.reduced:
            self.const_lp_term =\
                self.const_lp_term + np.log(self.degrees_of_freedom)
        self.metadata = metadata
    
    @property
    def reduced(self):
        """
        Property storing a boolean which determines whether this distribution
        represents a reduced chi squared statistic or not.
        """
        if not hasattr(self, '_reduced'):
            raise AttributeError("reduced referenced before it was set.")
        return self._reduced
    
    @reduced.setter
    def reduced(self, value):
        """
        Setter for the reduced property which determines whether this
        distribution represents a reduced chi squared statistic or not.
        
        value: boolean determining whether this distribution represents a
               reduced chi squared statistic or not.
        """
        if type(value) in bool_types:
            self._reduced = value
        else:
            raise TypeError("reduced was set to a non-bool.")
    
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
        sample = rand.chisquare(self.degrees_of_freedom, size=shape)
        if self.reduced:
            return sample / self.degrees_of_freedom
        else:
            return sample

    def log_value(self, point):
        """
        Evaluates and returns the log of the value of this distribution when
        the variable is point.
        
        point: numerical value of the variable
        """
        if self.reduced:
            point = point * self.degrees_of_freedom
        return self.const_lp_term - (point / 2) +\
            (((self.degrees_of_freedom / 2.) - 1) * np.log(point))
    
    def to_string(self):
        """
        Finds and returns the string representation of this
        ChiSquaredDistribution.
        """
        if self.reduced:
            return "ChiSquared({})".format(self.degrees_of_freedom)
        else:
            return "ReducedChiSquared({})".format(self.degrees_of_freedom)
    
    def __eq__(self, other):
        """
        Checks for equality between other and this object. Returns True if
        if other is a ChiSquaredDistribution with the same degrees_of_freedom.
        """
        if isinstance(other, ChiSquaredDistribution):
            dof_equal = (self.degrees_of_freedom == other.degrees_of_freedom)
            reduced_equal = (self.reduced == other.reduced)
            metadata_equal = self.metadata_equal(other)
            return all([dof_equal, reduced_equal, metadata_equal])
        else:
            return False
    
    def inverse_cdf(self, cdf):
        """
        Inverse of the cumulative distribution function.
        
        cdf: value between 0 and 1
        """
        answer = 2 * gammaincinv(self.degrees_of_freedom / 2, cdf)
        if self.reduced:
            answer = answer / self.degrees_of_freedom
        return answer
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data from this distribution. Only
        thing to save is the number of degrees of freedom.
        
        group: hdf5 file group to fill
        """
        group.attrs['class'] = 'ChiSquaredDistribution'
        group.attrs['degrees_of_freedom'] = self.degrees_of_freedom
        group.attrs['reduced'] = self.reduced
        self.save_metadata(group)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a ChiSquaredDistribution from the given hdf5 file group.
        
        group: the same hdf5 file group which fill_hdf5_group was called on
               when this Distribution was saved
        
        returns: a ChiSquaredDistribution object created from the information
                 in the given group
        """
        try:
            assert group.attrs['class'] == 'ChiSquaredDistribution'
        except:
            raise TypeError("The given hdf5 file doesn't seem to contain a " +\
                "ChiSquaredDistribution.")
        metadata = Distribution.load_metadata(group)
        degrees_of_freedom = group.attrs['degrees_of_freedom']
        reduced = group.attrs['reduced']
        return ChiSquaredDistribution(degrees_of_freedom, reduced=reduced,\
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
        
        point: single number at which to evaluate the deravitve
        
        returns: returns single number representing derivative of log value
        """
        constant = 1.
        if self.reduced:
            constant /= self.degrees_of_freedom
        return ((((self.degrees_of_freedom - 2) / point) - constant) / 2.)
    
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
        return ((2 - self.degrees_of_freedom) / (2. * (point ** 2)))

