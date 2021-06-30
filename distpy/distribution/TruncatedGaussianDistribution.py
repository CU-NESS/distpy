"""
Module containing class representing a truncated Gaussian distribution. Its PDF
is represented by: $$f(x) = \\sqrt{\\frac{2}{\\pi\\sigma^2}}\\ \
\\frac{e^{-(x-\\mu)^2/(2\\sigma^2)}}{\\text{erf}\\left(\\frac{x_{\\text{max}}-\
\\mu}{\\sqrt{2}\\sigma}\\right)-\\text{erf}\\left(\\frac{x_{\\text{min}}-\
\\mu}{\\sqrt{2}\\sigma}\\right)},$$ where
\\(x_{\\text{min}}\\le x\\le x_{\\text{max}}\\).

**File**: $DISTPY/distpy/distribution/TruncatedGaussianDistribution.py  
**Author**: Keith Tauscher  
**Date**: 31 May 2021
"""
from __future__ import division
import numpy as np
import numpy.random as rand
from scipy.special import erf, erfinv
from ..util import int_types, numerical_types
from .Distribution import Distribution

class TruncatedGaussianDistribution(Distribution):
    """
    Class representing a truncated Gaussian distribution. Its PDF is
    represented by: $$f(x) = \\sqrt{\\frac{2}{\\pi\\sigma^2}}\\ \
    \\frac{e^{-(x-\\mu)^2/(2\\sigma^2)}}{\\text{erf}\
    \\left(\\frac{x_{\\text{max}}-\\mu}{\\sqrt{2}\\sigma}\\right)-\
    \\text{erf}\\left(\\frac{x_{\\text{min}}-\\mu}{\\sqrt{2}\\sigma}\
    \\right)},$$ where \\(x_{\\text{min}}\\le x\\le x_{\\text{max}}\\).
    """
    def __init__(self, mean, variance, low=None, high=None, metadata=None):
        """
        Initializes a new `TruncatedGaussianDistribution` with the given
        parameter values.
        
        Parameters
        ----------
        mean : float
            mean, \\(\\mu\\), of **un-truncated** gaussian
        variance : float
            variance, \\(\\sigma^2\\) of **un-truncated** gaussian
        low : None or float
            the lowest point with nonzero probability density,
            \\(x_{\\text{min}}\\). `low` being None is the same as setting
            \\(x_{\\text{min}}\\rightarrow -\\infty\\)
        high : None or float
            the highest point with nonzero probability density,
            \\(x_{\\text{max}}\\). `high` being None is the same as setting
            \\(x_{\\text{max}}\\rightarrow +\\infty\\)
        metadata : number or str or dict or `distpy.util.Savable.Savable`
            data to store alongside this distribution.
        """
        self.internal_mean = mean
        self.internal_variance = variance
        self.low = low
        self.high = high
        self.metadata = metadata
    
    @property
    def low(self):
        """
        The lowest allowable value drawn from this distribution.
        """
        if not hasattr(self, '_low'):
            raise AttributeError("low was referenced before it was set.")
        return self._low
    
    @low.setter
    def low(self, value):
        """
        Setter for `TruncatedGaussianDistribution.low`.
        
        Parameters
        ----------
        value : float
            a real number
        """
        if type(value) is type(None):
            self._low = None
        elif type(value) in numerical_types:
            self._low = value
        else:
            raise ValueError("low was set to neither None nor a number.")
    
    @property
    def high(self):
        """
        The highest allowable value drawn from this distribution.
        """
        if not hasattr(self, '_high'):
            raise AttributeError("high was referenced before it was set.")
        return self._high
    
    @high.setter
    def high(self, value):
        """
        Setter for `TruncatedGaussianDistribution.high`.
        
        Parameters
        ----------
        value : float
            a real number larger than `TruncatedGaussianDistribution.low`
        """
        if type(value) is type(None):
            self._high = None
        elif type(value) in numerical_types:
            if value > self.low:
                self._high = value
            else:
                raise ValueError("high was set to a number less than or " +\
                    "equal to low.")
        else:
            raise ValueError("high was set to neither None nor a number.")
    
    @property
    def const_lp_term(self):
        """
        The constant part of the log probability density of this distribution.
        """
        if not hasattr(self, '_const_lp_term'):
            self._const_lp_term =\
                ((-1) * (np.log(np.pi * self.internal_variance / 2) / 2)) -\
                np.log(self.high_term - self.low_term)
        return self._const_lp_term
    
    @property
    def low_term(self):
        """
        The scaled error function at the low point.
        """
        if not hasattr(self, '_low_term'):
            if type(self.low) is type(None):
                self._low_term = -1
            else:
                self._low_term = erf((self.low - self.internal_mean) /\
                    np.sqrt(2 * self.internal_variance))
        return self._low_term
    
    @property
    def high_term(self):
        """
        The scaled error function at the high point.
        """
        if not hasattr(self, '_high_term'):
            if type(self.high) is type(None):
                self._high_term = 1
            else:
                self._high_term = erf((self.high - self.internal_mean) /\
                    np.sqrt(2 * self.internal_variance))
        return self._high_term
    
    @property
    def internal_mean(self):
        """
        The mean of the untruncated Gaussian used.
        """
        if not hasattr(self, '_internal_mean'):
            raise AttributeError("internal_mean was referenced before it " +\
                "was set.")
        return self._internal_mean
    
    @internal_mean.setter
    def internal_mean(self, value):
        """
        Setter for `TruncatedGaussianDistribution.internal_mean`.
        
        Parameters
        ----------
        value : float
            any real number
        """
        if type(value) in numerical_types:
            self._internal_mean = (value * 1.)
        else:
            raise TypeError("internal_mean was set to a non-number.")
    
    @property
    def internal_variance(self):
        """
        The variance of the untruncated Gaussian used.
        """
        if not hasattr(self, '_internal_variance'):
            raise AttributeError("internal_variance was referenced before " +\
                "it was set.")
        return self._internal_variance
    
    @internal_variance.setter
    def internal_variance(self, value):
        """
        Setter for `TruncatedGaussianDistribution.internal_variance`.
        
        Parameters
        ----------
        value : float
            any positive number
        """
        if type(value) in numerical_types:
            if value > 0:
                self._internal_variance = (value * 1.)
            else:
                raise ValueError("internal_variance must be positive.")
        else:
            raise TypeError("internal_variance was set to a non-number.")

    @property
    def numparams(self):
        """
        The number of parameters of this `TruncatedGaussianDistribution`, 1.
        """
        return 1
    
    @property
    def mean(self):
        """
        The mean of this `TruncatedGaussianDistribution`, which is different
        from \\(\\mu\\) in general because of the truncation.
        """
        if not hasattr(self, '_mean'):
            self._mean = self.internal_mean -\
                (np.sqrt(2 * self.internal_variance / np.pi) *\
                (np.exp(-(((self.high - self.internal_mean) /\
                np.sqrt(2 * self.internal_variance)) ** 2)) -\
                np.exp(-(((self.low - self.internal_mean) /\
                np.sqrt(2 * self.internal_variance)) ** 2))) /\
                (self.high_term - self.low_term))
        return self._mean
    
    @property
    def variance(self):
        """
        The variance of this `TruncatedGaussianDistribution`, which is
        different from \\(\\sigma^2\\) in general because of the truncation.
        """
        if not hasattr(self, '_variance'):
            variance = 1
            alpha = (self.low - self.internal_mean) /\
                np.sqrt(self.internal_variance)
            beta = (self.high - self.internal_mean) /\
                np.sqrt(self.internal_variance)
            phi_of_alpha = np.exp((alpha ** 2) / (-2)) / np.sqrt(2 * np.pi)
            phi_of_beta = np.exp((beta ** 2) / (-2)) / np.sqrt(2 * np.pi)
            denominator = (self.high_term - self.low_term) / 2
            variance = variance +\
                (((alpha * phi_of_alpha) - (beta * phi_of_beta)) / denominator)
            variance = variance -\
                (((phi_of_alpha - phi_of_beta) / denominator) ** 2)
            self._variance = variance * self.internal_variance
        return self._variance

    def draw(self, shape=None, random=rand):
        """
        Draws point(s) from this `TruncatedGaussianDistribution`.
        
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
        none_shape = (type(shape) is type(None))
        if none_shape:
            shape = (1,)
        elif type(shape) in int_types:
            shape = (shape,)
        unifs = random.rand(*shape)
        args_to_erfinv =\
            (unifs * self.high_term) + ((1. - unifs) * self.low_term)
        points = self.internal_mean +\
            (np.sqrt(2 * self.internal_variance) * erfinv(args_to_erfinv))
        if none_shape:
            return points[0]
        else:
            return points

    def log_value(self, point):
        """
        Computes the logarithm of the value of this
        `TruncatedGaussianDistribution` at the given point.
        
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
        if (type(self.low) is not type(None) and point < self.low) or\
                (type(self.high) is not type(None) and point > self.high):
            return -np.inf
        return (self.const_lp_term -\
            ((point - self.internal_mean) ** 2) / (2 * self.internal_variance))

    def to_string(self):
        """
        Finds and returns a string version of this
        `TruncatedGaussianDistribution` of the form
        `"Normal(mu, sigma2) on [low,high]"`.
        """
        if type(self.low) is type(None):
            low_string = "-inf"
        else:
            low_string = "{:.1g}".format(self.low)
        if type(self.high) is type(None):
            high_string = "inf"
        else:
            high_string = "{:.1g}".format(self.high)
        return "Normal({0:.2g}, {1:.2g}) on [{2!s},{3!s}]".format(\
            self.internal_mean, self.internal_variance, low_string,\
            high_string)
    
    def __eq__(self, other):
        """
        Checks for equality of this `TruncatedGaussianDistribution` with
        `other`.
        
        Parameters
        ----------
        other : object
            object to check for equality
        
        Returns
        -------
        result : bool
            True if and only if `other` is a `TruncatedGaussianDistribution`
            with the same internal mean and variance and bounds
        """
        if isinstance(other, TruncatedGaussianDistribution):
            mean_close = np.isclose(self.internal_mean, other.internal_mean,\
                rtol=0, atol=1e-9)
            variance_close = np.isclose(self.internal_variance,\
                other.internal_variance, rtol=1e-12, atol=0)
            if type(self.high) is type(None):
                hi_close = (type(other.high) is type(None))
            elif type(other.high) is not type(None):
                hi_close = np.isclose(self.high, other.high, rtol=0, atol=1e-9)
            else:
                # since self.high is not None in this block, just return False
                return False
            if type(self.low) is type(None):
                lo_close = (type(other.low) is type(None))
            elif type(other.low) is not type(None):
                lo_close = np.isclose(self.low, other.low, rtol=0, atol=1e-9)
            else:
                return False
            metadata_equal = self.metadata_equal(other)
            return all([mean_close, variance_close, hi_close, lo_close,\
                metadata_equal])
        else:
            return False
    
    def inverse_cdf(self, cdf):
        """
        Computes the inverse of the cumulative distribution function (cdf) of
        this `TruncatedGaussianDistribution`.
        
        Parameters
        ----------
        cdf : float
            probability value between 0 and 1
        
        Returns
        -------
        point : float
            value which yields `cdf` when it the CDF is evaluated at it
        """
        erfinv_args =\
            (self.low_term + (cdf * (self.high_term - self.low_term)))
        return (self.internal_mean +\
            (np.sqrt(2 * self.internal_variance) * erfinv(erfinv_args)))
    
    @property
    def minimum(self):
        """
        The minimum allowable value(s) in this distribution.
        """
        return self.low
    
    @property
    def maximum(self):
        """
        The maximum allowable value(s) in this distribution.
        """
        return self.high
    
    @property
    def is_discrete(self):
        """
        Boolean describing whether this distribution is discrete (True) or
        continuous (False).
        """
        return False
    
    def fill_hdf5_group(self, group, save_metadata=True):
        """
        Fills the given hdf5 file group with data about this
        `TruncatedGaussianDistribution` so that it can be loaded later.
        
        Parameters
        ----------
        group : h5py.Group
            hdf5 file group to fill
        save_metadata : bool
            - if True, attempts to save metadata alongside distribution and
            throws error if it fails
            - if False, metadata is ignored in saving process
        """
        group.attrs['class'] = 'TruncatedGaussianDistribution'
        if type(self.low) is not type(None):
            group.attrs['low'] = self.low
        if type(self.high) is not type(None):
            group.attrs['high'] = self.high
        group.attrs['mean'] = self.internal_mean
        group.attrs['variance'] = self.internal_variance
        if save_metadata:
            self.save_metadata(group)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a `TruncatedGaussianDistribution` from the given hdf5 file group.
        
        Parameters
        ----------
        group : h5py.Group
            the same hdf5 file group which fill_hdf5_group was called on when
            this Distribution was saved
        
        Returns
        -------
        distribution : `TruncatedGaussianDistribution`
            distribution created from the information in the given group
        """
        try:
            assert group.attrs['class'] == 'TruncatedGaussianDistribution'
        except:
            raise TypeError("The given hdf5 file doesn't seem to contain a " +\
                "TruncatedGaussianDistribution.")
        metadata = Distribution.load_metadata(group)
        mean = group.attrs['mean']
        variance = group.attrs['variance']
        if 'low' in group.attrs:
            low = group.attrs['low']
        else:
            low = None
        if 'high' in group.attrs:
            high = group.attrs['high']
        else:
            high = None
        return TruncatedGaussianDistribution(mean, variance, low=low,\
            high=high, metadata=metadata)
    
    @property
    def gradient_computable(self):
        """
        Boolean describing whether the gradient of the given distribution has
        been implemented. If True,
        `TruncatedGaussianDistribution.gradient_of_log_value` method can be
        called safely.
        """
        return True
    
    def gradient_of_log_value(self, point):
        """
        Computes the gradient (derivative) of the logarithm of the value of
        this `TruncatedGaussianDistribution` at the given point.
        
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
        return (self.internal_mean - point) / self.internal_variance
    
    @property
    def hessian_computable(self):
        """
        Boolean describing whether the hessian of the given distribution has
        been implemented. If True,
        `TruncatedGaussianDistribution.hessian_of_log_value` method can be
        called safely.
        """
        return True
    
    def hessian_of_log_value(self, point):
        """
        Computes the hessian (second derivative) of the logarithm of the value
        of this `TruncatedGaussianDistribution` at the given point.
        
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
        return (-1.) / self.internal_variance
    
    def copy(self):
        """
        Copies this distribution.
        
        Returns
        -------
        copied : `TruncatedGaussianDistribution`
            a deep copy of this distribution, ignoring metadata.
        """
        return TruncatedGaussianDistribution(self.internal_mean,\
            self.internal_variance, self.low, self.high)

