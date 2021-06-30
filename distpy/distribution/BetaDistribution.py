"""
Module containing class representing a beta distribution. Its PDF is
represented by:
$$f(x) = \\frac{x^{\\alpha-1}(1-x)^{\\beta-1}}{B(\\alpha,\\beta)},$$ where
\\(B(x,y)\\) is the beta function.

**File**: $DISTPY/distpy/distribution/BetaDistribution.py  
**Author**: Keith Tauscher  
**Date**: 31 May 2021
"""
from __future__ import division
import numpy as np
import numpy.random as rand
from scipy.special import beta as beta_func
from scipy.special import betaincinv
from ..util import numerical_types
from .Distribution import Distribution

class BetaDistribution(Distribution):
    """
    Class representing a beta distribution. Its PDF is represented by:
    $$f(x) = \\frac{x^{\\alpha-1}(1-x)^{\\beta-1}}{B(\\alpha,\\beta)},$$ where
    \\(B(x,y)\\) is the beta function.
    """
    def __init__(self, alpha, beta, metadata=None):
        """
        Initializes a new `BetaDistribution` with the given parameter values.
        
        Parameters
        ----------
        alpha : float
            real number, \\(\\alpha>0\\)
        beta : float
            real number, \\(\\beta>0\\)
        metadata : number or str or dict or `distpy.util.Savable.Savable`
            data to store alongside this distribution.
        """
        self.alpha = alpha
        self.beta = beta
        self.metadata = metadata
    
    @staticmethod
    def create_from_mean_and_variance(mean, variance, metadata=None):
        """
        Creates a new `BetaDistribution` with the given mean and variance.
        
        Parameters
        ----------
        mean : float
            the mean of the desired distribution, must satisfy `0<mean<1`
        variance : float
            the variance of the desired distribution, must satisfy
            `0<variance<(mean*(1-mean))`
        metadata : number or str or dict or `distpy.util.Savable.Savable`
            data to store alongside this distribution
        """
        if (mean <= 0) or (mean >= 1):
            raise ValueError("The mean of a BetaDistribution must be " +\
                "between 0 and 1 (exclusive)")
        if (variance <= 0) or (variance >= (mean * (1 - mean))):
            raise ValueError("The variance of a BetaDistribution must be " +\
                "less than mean*(1-mean).")
        alpha_plus_beta = (((mean * (1 - mean)) / variance) - 1)
        alpha = (mean * alpha_plus_beta)
        beta = ((1 - mean) * alpha_plus_beta)
        return BetaDistribution(alpha, beta, metadata=metadata)
    
    @property
    def alpha(self):
        """
        The parameter, \\(\\alpha\\), of this distribution.
        """
        if not hasattr(self, '_alpha'):
            raise AttributeError("alpha was referenced before it was set.")
        return self._alpha
    
    @alpha.setter
    def alpha(self, value):
        """
        Setter for `BetaDistribution.beta`.
        
        Parameters
        ----------
        value : float
            positive number
        """
        if type(value) in numerical_types:
            if value > 0:
                self._alpha = value
            else:
                raise ValueError("alpha was set to a non-positive number.")
        else:
            raise TypeError("alpha was set to a non-number.")
    
    @property
    def alpha_minus_one(self):
        """
        \\(alpha-1\\)
        """
        if not hasattr(self, '_alpha_minus_one'):
            self._alpha_minus_one = self.alpha - 1
        return self._alpha_minus_one
    
    @property
    def beta(self):
        """
        The parameter, \\(\\beta\\), of this distribution.
        """
        if not hasattr(self, '_beta'):
            raise AttributeError("beta was referenced before it was set.")
        return self._beta
    
    @beta.setter
    def beta(self, value):
        """
        Setter for `BetaDistribution.beta`
        
        Parameters
        ----------
        value : float
            positive number
        """
        if type(value) in numerical_types:
            if value > 0:
                self._beta = value
            else:
                raise ValueError("beta was set to a non-positive number.")
        else:
            raise TypeError("beta was set to a non-number.")
    
    @property
    def beta_minus_one(self):
        """
        \\(\\beta-1\\)
        """
        if not hasattr(self, '_beta_minus_one'):
            self._beta_minus_one = self.beta - 1
        return self._beta_minus_one
    
    @property
    def mean(self):
        """
        The mean of this `BetaDistribution`,
        \\(\\frac{\\alpha}{\\alpha+\\beta}\\).
        """
        if not hasattr(self, '_mean'):
            self._mean = self.alpha / (self.alpha + self.beta)
        return self._mean
    
    @property
    def variance(self):
        """
        The variance of this `BetaDistribution`,
        \\(\\frac{\\alpha\\beta}{(\\alpha+\\beta)^2(\\alpha+\\beta+1)}\\).
        """
        if not hasattr(self, '_variance'):
            self._variance = (self.alpha * self.beta) /\
                (((self.alpha + self.beta) ** 2) *\
                (self.alpha + self.beta + 1))
        return self._variance
    
    @property
    def const_lp_term(self):
        """
        The constant term in the log of the pdf of this distribution, given by
        \\(-\\ln{B(\\alpha,\\beta)}\\) where \\(B(x,y)\\) is the Beta function.
        """
        if not hasattr(self, '_const_lp_term'):
            self._const_lp_term = -np.log(beta_func(self.alpha, self.beta))
        return self._const_lp_term
    
    @property
    def numparams(self):
        """
        The number of parameters of this `BetaDistribution`, 1.
        """
        return 1
    
    def draw(self, shape=None, random=rand):
        """
        Draws point(s) from this `BetaDistribution`.
        
        Parameters
        ----------
        shape : int or tuple or None
            - if None, returns single random variate as a scalar
            - if int, \\(n\\), returns \\(n\\) random variates in a 1D array of
            length \\(n\\)
            - if tuple of \\(n\\) ints, returns `numpy.prod(shape)` random
            variates as an \\(n\\)-D array of shape `shape` is returned
        random : `numpy.random.RandomState`
            the random number generator to use (by default, `numpy.random` is
            used)
        
        Returns
        -------
        variates : float or `numpy.ndarray`
            either single random variates or array of such variates. See
            documentation of `shape` above for type and shape of return value
        """
        return random.beta(self.alpha, self.beta, size=shape)
    
    def log_value(self, point):
        """
        Computes the logarithm of the value of this `BetaDistribution` at the
        given point.
        
        Parameters
        ----------
        point : float
            scalar at which to evaluate PDF
        
        Returns
        -------
        value : float
            natural logarithm of the value of this distribution at `point`. If
            \\(f\\) is this distribution's PDF and \\(x\\) is `point`, then
            `value` is \\(\\ln{\\big(f(x)\\big)}\\)
        """
        if (point <= 0) or (point >= 1):
            return -np.inf
        return self.const_lp_term + (self.alpha_minus_one * np.log(point)) +\
               (self.beta_minus_one * np.log(1. - point))
    
    def to_string(self):
        """
        Finds and returns a string version of this `BetaDistribution` of the
        form `"Beta(alpha, beta)"`.
        """
        return "Beta({0:.2g}, {1:.2g})".format(self.alpha, self.beta)
    
    def __eq__(self, other):
        """
        Checks for equality of this `BetaDistribution` with `other`.
        
        Parameters
        ----------
        other : object
            object to check for equality
        
        Returns
        -------
        result : bool
            True if and only if `other` is a `BetaDistribution` with the same
            `BetaDistribution.alpha` and `BetaDistribution.beta`
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
    def minimum(self):
        """
        The minimum allowable value(s) in this distribution.
        """
        return 0
    
    @property
    def maximum(self):
        """
        The maximum allowable value(s) in this distribution.
        """
        return 1
    
    @property
    def is_discrete(self):
        """
        Boolean describing whether this distribution is discrete (True) or
        continuous (False).
        """
        return False
    
    def fill_hdf5_group(self, group, save_metadata=True):
        """
        Fills the given hdf5 file group with data about this `BetaDistribution`
        so that it can be loaded later.
        
        Parameters
        ----------
        group : h5py.Group
            hdf5 file group to fill
        save_metadata : bool
            - if True, attempts to save metadata alongside distribution and
            throws error if it fails
            - if False, metadata is ignored in saving process
        """
        group.attrs['class'] = 'BetaDistribution'
        group.attrs['alpha'] = self.alpha
        group.attrs['beta'] = self.beta
        if save_metadata:
            self.save_metadata(group)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a `BetaDistribution` from the given hdf5 file group.
        
        Parameters
        ----------
        group : h5py.Group
            the same hdf5 file group which fill_hdf5_group was called on when
            this Distribution was saved
        
        Returns
        -------
        distribution : `BetaDistribution`
            distribution created from the information in the given group
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
        Computes the inverse of the cumulative distribution function (cdf) of
        this `BetaDistribution`.
        
        Parameters
        ----------
        cdf : float
            probability value between 0 and 1
        
        Returns
        -------
        point : float
            value which yields `cdf` when it the CDF is evaluated at it
        """
        return betaincinv(self.alpha, self.beta, cdf)
    
    @property
    def gradient_computable(self):
        """
        Boolean describing whether the gradient of the given distribution has
        been implemented. If True, `BetaDistribution.gradient_of_log_value`
        method can be called safely.
        """
        return True
    
    def gradient_of_log_value(self, point):
        """
        Computes the gradient (derivative) of the logarithm of the value of
        this `BetaDistribution` at the given point.
        
        Parameters
        ----------
        point : float
            scalar at which to evaluate the gradient
        
        Returns
        -------
        value : float
            gradient of the natural logarithm of the value of this
            distribution. If \\(f\\) is this distribution's PDF and \\(x\\) is
            `point`, then `value` is
            \\(\\boldsymbol{\\nabla}\\ln{\\big(f(x)\\big)}\\) as a float
        """
        return (((self.alpha - 1) / point) - ((self.beta - 1) / (1 - point)))
    
    @property
    def hessian_computable(self):
        """
        Boolean describing whether the hessian of the given distribution has
        been implemented. If True, `BetaDistribution.hessian_of_log_value`
        method can be called safely.
        """
        return True
    
    def hessian_of_log_value(self, point):
        """
        Computes the hessian (second derivative) of the logarithm of the value
        of this `BetaDistribution` at the given point.
        
        Parameters
        ----------
        point : float
            scalar at which to evaluate the gradient
        
        Returns
        -------
        value : float
            hessian of the natural logarithm of the value of this
            distribution. If \\(f\\) is this distribution's PDF and \\(x\\) is
            `point`, then `value` is
            \\(\\boldsymbol{\\nabla}\\boldsymbol{\\nabla}^T\
            \\ln{\\big(f(x)\\big)}\\) as a float
        """
        return (-(((self.alpha - 1) / (point ** 2)) +\
            ((self.beta - 1) / ((1 - point) ** 2))))
    
    def copy(self):
        """
        Copies this distribution.
        
        Returns
        -------
        copied : `BetaDistribution`
            a deep copy of this distribution, ignoring metadata.
        """
        return BetaDistribution(self.alpha, self.beta)

