"""
File: distpy/DoubleSidedExponentialDistribution.py
Author: Keith Tauscher
Date: 6 Aug 2017

Description: File containing class representing double sided exponential
             distribution.
"""
from __future__ import division
import numpy as np
import numpy.random as rand
from ..util import bool_types, numerical_types
from .Distribution import Distribution

class DoubleSidedExponentialDistribution(Distribution):
    """
    Distribution with a double-sided exponential distribution. Double sided
    exponential distributions are "peak"ier than Gaussians.
    """
    def __init__(self, mean, variance, metadata=None):
        """
        Initializes a new DoubleSidedExponentialDistribution with the given
        parameters.
        
        mean: mean, mode and median of the distribution
        variance: variance of distribution
        """
        if type(mean) in numerical_types:
            self.mean = (mean * 1.)
        else:
            raise ValueError('The mean parameter given to a ' +\
                'DoubleSidedExponentialDistribution was not of a numerical ' +\
                'type.')
        if type(variance) in numerical_types:
            if variance > 0:
                self.variance = (1. * variance)
            else:
                raise ValueError("The variance given to a " +\
                    "DoubleSidedExponentialDistribution was not positive.")
        else:
            raise ValueError("The variance given to a " +\
                "DoubleSidedExponentialDistribution was not of a numerical " +\
                "type.")
        self._const_lp_term = (np.log(2) + np.log(self.variance)) / (-2)
        self.metadata = metadata
    
    @property
    def numparams(self):
        """
        Exponential pdf is univariate so numparams always returns 1.
        """
        return 1
    
    @property
    def root_half_variance(self):
        """
        Property storing the square root of half the variance. This is the same
        as 1/sqrt(2) times the standard deviation.
        """
        if not hasattr(self, '_root_half_variance'):
            self._root_half_variance = np.sqrt(self.variance / 2.)
        return self._root_half_variance
    
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
        return rand.laplace(loc=self.mean, scale=self.root_half_variance,\
            size=shape)
    
    def log_value(self, point):
        """
        Evaluates and returns the log of the value of this distribution when
        the variable is value.
        
        point: numerical value of the variable
        """
        return self._const_lp_term -\
            (np.abs(point - self.mean) / self.root_half_variance)

    def to_string(self):
        """
        Finds and returns a string version of this
        DoubleSidedExponentialDistribution.
        """
        return "DSExp({0:.2g}, {1:.2g})".format(self.mean, self.variance)
    
    def __eq__(self, other):
        """
        Checks for equality of this distribution with other. Returns True if
        other is an DoubleSidedExponentialDistribution with the same mean and
        sigma and False otherwise.
        """
        if isinstance(other, DoubleSidedExponentialDistribution):
            tol_kwargs = {'rtol': 1e-6, 'atol': 1e-9}
            mean_equal = np.isclose(self.mean, other.mean, **tol_kwargs)
            variance_equal =\
                np.isclose(self.variance, other.variance, **tol_kwargs)
            metadata_equal = self.metadata_equal(other)
            return all([mean_equal, variance_equal, metadata_equal])
        else:
            return False
    
    def inverse_cdf(self, cdf):
        """
        Inverse of the cumulative distribution function.
        
        cdf: value between 0 and 1
        """
        twice_distances_from_mean_cdf = np.abs((2 * cdf) - 1)
        distances_from_mean = ((-self.root_half_variance) *\
            np.log(1 - twice_distances_from_mean_cdf))
        on_right = ((2 * cdf) > 1)
        if type(on_right) in bool_types:
            # cdf is a single value here!
            multiplicative_displacements = ((2 * int(on_right)) - 1)
        else:
            # cdf is an array here!
            multiplicative_displacements = ((2 * on_right.astype(int)) - 1)
        return\
            (self.mean + (multiplicative_displacements * distances_from_mean))
        
        
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data about this distribution. The
        only things to save are the class name, mean, and variance.
        
        group: hdf5 file group to fill
        """
        group.attrs['class'] = 'DoubleSidedExponentialDistribution'
        group.attrs['mean'] = self.mean
        group.attrs['variance'] = self.variance
        self.save_metadata(group)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a DoubleSidedExponentialDistribution from the given hdf5 file
        group.
        
        group: the same hdf5 file group which fill_hdf5_group was called on
               when this Distribution was saved
        
        returns: a ExponentialDistribution object created from the information
                 in the given group
        """
        try:
            assert group.attrs['class'] == 'DoubleSidedExponentialDistribution'
        except:
            raise TypeError("The given hdf5 file doesn't seem to contain a " +\
                "DoubleSidedExponentialDistribution.")
        metadata = Distribution.load_metadata(group)
        mean = group.attrs['mean']
        variance = group.attrs['variance']
        return DoubleSidedExponentialDistribution(mean, variance,\
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
        return np.sign(self.mean - point) / self.root_half_variance
    
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
        return 0

