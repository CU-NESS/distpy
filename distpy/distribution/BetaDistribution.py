"""
File: distpy/distribution/BetaDistribution.py
Author: Keith Tauscher
Date: 12 Feb 2018

Description: File containing class representing a beta distribution.
"""
import numpy as np
import numpy.random as rand
from scipy.special import beta as beta_func
from scipy.special import betaincinv
from ..util import numerical_types
from .Distribution import Distribution

class BetaDistribution(Distribution):
    """
    Class representing a beta distribution. Useful for parameters which must
    lie between 0 and 1. Classically, this is the distribution of the
    probability of success in a binary experiment where alpha successes and
    beta failures have been observed.
    """
    def __init__(self, alpha, beta, metadata=None):
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
                raise ValueError('The alpha or beta parameter given to a ' +\
                    'Beta was not non-negative.')
        else:
            raise ValueError('The alpha or beta parameter given to a Beta ' +\
                'were not of a numerical type.')
        self.const_lp_term = -np.log(beta_func(self.alpha, self.beta))
        self.metadata = metadata
    
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
        return "Beta({0:.2g}, {1:.2g})".format(self.alpha, self.beta)
    
    def __eq__(self, other):
        """
        Checks for equality of this object with other. Returns True if other is
        a BetaDistribution with nearly the same alpha and beta (down to 10^-9
        level) and False otherwise.
        """
        if isinstance(other, BetaDistribution):
            tol_kwargs = {'rtol': 0., 'atol': 1e-9}
            alpha_equal = np.isclose(self.alpha, other.alpha, **tol_kwargs)
            beta_equal = np.isclose(self.beta, other.beta, **tol_kwargs)
            metadata_equal = self.metadata_equal(other)
            return all([alpha_equal, beta_equal, metadata_equal])
        else:
            return False
    
    @property
    def is_discrete(self):
        """
        Property storing a boolean describing whether this distribution is
        discrete (True) or continuous (False).
        """
        return False
    
    def fill_hdf5_group(self, group, save_metadata=True):
        """
        Fills the given hdf5 group with data from this distribution. All that
        is to be saved is the class name, alpha, and beta.
        
        group: hdf5 file group to fill
        save_metadata: if True, attempts to save metadata alongside
                                distribution and throws error if it fails
                       if False, metadata is ignored in saving process
        """
        group.attrs['class'] = 'BetaDistribution'
        group.attrs['alpha'] = self.alpha
        group.attrs['beta'] = self.beta
        if save_metadata:
            self.save_metadata(group)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a BetaDistribution from the given hdf5 file group.
        
        group: the same hdf5 file group which fill_hdf5_group was called on
               when this Distribution was saved
        
        returns: a BetaDistribution object created from the information in the
                 given group
        """
        try:
            assert group.attrs['class'] == 'BetaDistribution'
        except:
            raise TypeError("The given hdf5 file doesn't seem to contain a " +\
                "BetaDistribution.")
        metadata = Distribution.load_metadata(group)
        alpha = group.attrs['alpha']
        beta = group.attrs['beta']
        return BetaDistribution(alpha, beta, metadata=metadata)
    
    def inverse_cdf(self, cdf):
        """
        Inverse of the cumulative distribution function (cdf) of this
        distribution.
        
        cdf: value between 0 and 1
        """
        return betaincinv(self.alpha, self.beta, cdf)
    
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
        return (((self.alpha - 1) / point) - ((self.beta - 1) / (1 - point)))
    
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
        return (-(((self.alpha - 1) / (point ** 2)) +\
            ((self.beta - 1) / ((1 - point) ** 2))))

