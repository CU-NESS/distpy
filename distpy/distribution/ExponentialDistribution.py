"""
Module containing class representing a exponential distribution. Its PDF is
represented by: $$f(x)=\\begin{cases}\\frac{1}{\\theta}e^{-(x-\\mu)/\\theta} &\
x\\ge \\mu \\\\ 0 & \\text{otherwise} \\end{cases}$$

**File**: $DISTPY/distpy/distribution/ExponentialDistribution.py  
**Author**: Keith Tauscher  
**Date**: 31 May 2021
"""
from __future__ import division
import numpy as np
import numpy.random as rand
from ..util import numerical_types
from .Distribution import Distribution

class ExponentialDistribution(Distribution):
    """
    Class representing a exponential distribution. Its PDF is represented by:
    $$f(x)=\\begin{cases}\\frac{1}{\\theta}e^{-(x-\\mu)/\\theta} &\
    x\\ge \\mu \\\\ 0 & \\text{otherwise} \\end{cases}$$
    """
    def __init__(self, rate, shift=0., metadata=None):
        """
        Initializes a new `ExponentialDistribution` with the given parameter
        values.
        
        Parameters
        ----------
        rate : float
            positive real number, equal to \\(1/\\theta\\)
        shift : float
            lowest point, \\(\\mu\\), with nonzero probability density
        metadata : number or str or dict or `distpy.util.Savable.Savable`
            data to store alongside this distribution.
        """
        self.rate = rate
        self.shift = shift
        self.metadata = metadata
    
    @property
    def rate(self):
        """
        The rate parameter, \\(\\lambda\\) of this distribution.
        """
        if not hasattr(self, '_rate'):
            raise AttributeError("rate was referenced before it was set.")
        return self._rate
    
    @rate.setter
    def rate(self, value):
        """
        Setter for `ExponentialDistribution.rate`.
        
        Parameters
        ----------
        value : float
            single positive number
        """
        if type(value) in numerical_types:
            if value > 0:
                self._rate = value
            else:
                raise ValueError("rate property of ExponentialDistribution " +\
                    "was set to a non-positive number.")
        else:
            raise TypeError("rate property of ExponentialDistribution was " +\
                "set to a non-number.")
    
    @property
    def shift(self):
        """
        The shift parameter, \\(\\mu\\), which is the left endpoint of this
        distribution's domain of support.
        """
        if not hasattr(self, '_shift'):
            raise AttributeError("shift was referenced before it was set.")
        return self._shift
    
    @shift.setter
    def shift(self, value):
        """
        Setter for `ExponentialDistribution.shift`.
        
        Parameters
        ----------
        value : float
            any real number
        """
        if type(value) in numerical_types:
            self._shift = value
        else:
            raise TypeError("shift property of ExponentialDistribution was " +\
                "set to a non-number.")
    
    @staticmethod
    def create_from_mean_and_variance(mean, variance, metadata=None):
        """
        Creates a new `ExponentialDistribution` from the mean and variance
        instead of the rate and shift parameters.
        
        Parameters
        ----------
        mean : float
            real number equal to the mean of the desired distribution
        variance : float
            positive number equal to the variance of the desired distribution
        metadata : number or str or dict or `distpy.util.Savable.Savable`
            data to store alongside this distribution.
        
        Returns
        -------
        distribution : `ExponentialDistribution`
            `ExponentialDistribution` with the given mean and variance
        """
        standard_deviation = np.sqrt(variance)
        rate = (1 / standard_deviation)
        shift = (mean - standard_deviation)
        return ExponentialDistribution(rate, shift=shift, metadata=None)
    
    @property
    def numparams(self):
        """
        The number of parameters of this `ExponentialDistribution`, 1.
        """
        return 1
    
    @property
    def mean(self):
        """
        The mean of this `ExponentialDistribution`, \\(\\mu + \\theta\\).
        """
        if not hasattr(self, '_mean'):
            self._mean = self.shift + (1 / self.rate)
        return self._mean
    
    @property
    def variance(self):
        """
        The variance of this `ExponentialDistribution`, \\(\\theta^2\\).
        """
        if not hasattr(self, '_variance'):
            self._variance = (1 / (self.rate ** 2))
        return self._variance
    
    def draw(self, shape=None, random=rand):
        """
        Draws point(s) from this `ExponentialDistribution`.
        
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
        return\
            random.exponential(scale=(1 / self.rate), size=shape) + self.shift
    
    def log_value(self, point):
        """
        Computes the logarithm of the value of this `ExponentialDistribution`
        at the given point.
        
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
        val_min_shift = point - self.shift
        if val_min_shift < 0:
            return -np.inf
        return (np.log(self.rate) - (self.rate * val_min_shift))

    def to_string(self):
        """
        Finds and returns a string version of this `ExponentialDistribution` of
        the form `"Exp(1/theta, shift=\\mu)"`.
        """
        if self.shift != 0:
            return "Exp({0:.2g}, shift={1:.2g})".format(self.rate, self.shift)
        return "Exp({:.2g})".format(self.rate)
    
    def __eq__(self, other):
        """
        Checks for equality of this `ExponentialDistribution` with `other`.
        
        Parameters
        ----------
        other : object
            object to check for equality
        
        Returns
        -------
        result : bool
            True if and only if `other` is a `ExponentialDistribution` with the
            same `ExponentialDistribution.rate` and
            `ExponentialDistribution.shift`
        """
        if isinstance(other, ExponentialDistribution):
            rate_close = np.isclose(self.rate, other.rate, rtol=1e-9, atol=0)
            shift_close =\
                np.isclose(self.shift, other.shift, rtol=0, atol=1e-9)
            metadata_equal = self.metadata_equal(other)
            return all([rate_close, shift_close, metadata_equal])
        else:
            return False
    
    def inverse_cdf(self, cdf):
        """
        Computes the inverse of the cumulative distribution function (cdf) of
        this `ExponentialDistribution`.
        
        Parameters
        ----------
        cdf : float
            probability value between 0 and 1
        
        Returns
        -------
        point : float
            value which yields `cdf` when it the CDF is evaluated at it
        """
        return (self.shift - (np.log(1 - cdf) / self.rate))
    
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
        return None
    
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
        `ExponentialDistribution` so that it can be loaded later.
        
        Parameters
        ----------
        group : h5py.Group
            hdf5 file group to fill
        save_metadata : bool
            - if True, attempts to save metadata alongside distribution and
            throws error if it fails
            - if False, metadata is ignored in saving process
        """
        group.attrs['class'] = 'ExponentialDistribution'
        group.attrs['rate'] = self.rate
        group.attrs['shift'] = self.shift
        if save_metadata:
            self.save_metadata(group)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a `ExponentialDistribution` from the given hdf5 file group.
        
        Parameters
        ----------
        group : h5py.Group
            the same hdf5 file group which fill_hdf5_group was called on when
            this Distribution was saved
        
        Returns
        -------
        distribution : `ExponentialDistribution`
            distribution created from the information in the given group
        """
        try:
            assert group.attrs['class'] == 'ExponentialDistribution'
        except:
            raise TypeError("The given hdf5 file doesn't seem to contain a " +\
                "ExponentialDistribution.")
        metadata = Distribution.load_metadata(group)
        rate = group.attrs['rate']
        shift = group.attrs['shift']
        return ExponentialDistribution(rate, shift=shift, metadata=metadata)
    
    @property
    def gradient_computable(self):
        """
        Boolean describing whether the gradient of the given distribution has
        been implemented. If True,
        `ExponentialDistribution.gradient_of_log_value` method can be called
        safely.
        """
        return True
    
    def gradient_of_log_value(self, point):
        """
        Computes the gradient (derivative) of the logarithm of the value of
        this `ExponentialDistribution` at the given point.
        
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
        return -self.rate
    
    @property
    def hessian_computable(self):
        """
        Boolean describing whether the hessian of the given distribution has
        been implemented. If True,
        `ExponentialDistribution.hessian_of_log_value` method can be called
        safely.
        """
        return True
    
    def hessian_of_log_value(self, point):
        """
        Computes the hessian (second derivative) of the logarithm of the value
        of this `ExponentialDistribution` at the given point.
        
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
        return 0.
    
    def copy(self):
        """
        Copies this distribution.
        
        Returns
        -------
        copied : `ExponentialDistribution`
            a deep copy of this distribution, ignoring metadata.
        """
        return ExponentialDistribution(self.rate, self.shift)

