"""
Module containing class representing a Gamma distribution. Its PDF is
represented by: $$f(x) = \\frac{1}{\\Gamma(k/p)}\\ \\frac{p}{\\theta}\\ \
\\left(\\frac{x}{\\theta}\\right)^{k-1}\\ e^{-(x/\\theta)^p},$$ where
\\(\\Gamma(x)\\) is the Gamma function.

**File**: $DISTPY/distpy/distribution/GammaDistribution.py  
**Author**: Keith Tauscher  
**Date**: 31 May 2021
"""
from __future__ import division
import numpy as np
import numpy.random as rand
from scipy.special import gamma, gammaln, gammaincinv
from ..util import numerical_types
from .Distribution import Distribution

class GammaDistribution(Distribution):
    """
    Class representing a Gamma distribution. Its PDF is represented by:
    $$f(x) = \\frac{1}{\\Gamma(k/p)}\\ \\frac{p}{\\theta}\\ \
    \\left(\\frac{x}{\\theta}\\right)^{k-1} \\ e^{-(x/\\theta)^p},$$ where
    \\(\\Gamma(x)\\) is the Gamma function.
    """
    def __init__(self, shape, scale=1, power=1, metadata=None):
        """
        Initializes a new `GammaDistribution` with the given parameter values.
        
        Parameters
        ----------
        shape : float
            positive real number, \\(k\\)
        scale : float
            positive real number, \\(\\theta\\)
        power : float
            positive real number, \\(p\\)
        metadata : number or str or dict or `distpy.util.Savable.Savable`
            data to store alongside this distribution.
        """
        self.shape = shape
        self.scale = scale
        self.power = power
        self.metadata = metadata
    
    @staticmethod
    def create_from_mean_and_variance(mean, variance, metadata=None):
        """
        Creates a new `GammaDistribution` from the mean and variance (assuming
        power=1) instead of through the conventional shape and scale
        parameters.
        
        Parameters
        ----------
        mean : float
            positive number equal to mean of desired distribution
        variance : float
            positive number equal to variance of desired distribution
        metadata : number or str or dict or `distpy.util.Savable.Savable`
            data to store alongside this distribution.
        
        Returns
        -------
        distribution : `GammaDistribution`
            new GammaDistribution object encoding the desired distribution
        """
        shape = ((mean ** 2) / variance)
        scale = (variance / mean)
        return\
            GammaDistribution(shape, scale=scale, power=1, metadata=metadata)
    
    @property
    def shape(self):
        """
        The shape parameter, \\(k\\), of this `GammaDistribution`. This is
        the power of \\(x\\) in the product in the pdf (not the power of
        \\(x\\) in the exponent of the pdf).
        """
        if not hasattr(self, '_shape'):
            raise AttributeError("shape was referenced before it was set.")
        return self._shape
    
    @property
    def shape_minus_one(self):
        """
        \\(k-1\\)
        """
        if not hasattr(self, '_shape_minus_one'):
            self._shape_minus_one = self.shape - 1
        return self._shape_minus_one
    
    @shape.setter
    def shape(self, value):
        """
        Setter for `GammaDistribution.shape`.
        
        Parameters
        ----------
        value : float
            a positive number
        """
        if type(value) in numerical_types:
            if value > 0:
                self._shape = value
            else:
                raise ValueError("shape given to GammaDistribution was not " +\
                    "positive.")
        else:
            raise TypeError("shape given to GammaDistribution was not a " +\
                "number.")
    
    @property
    def scale(self):
        """
        The scale parameter, \\(\\theta\\) of the gamma distribution. This is
        what \\(x\\) is divided by when it appears in the pdf.
        """
        if not hasattr(self, '_scale'):
            raise AttributeError("scale was referenced before it was set.")
        return self._scale
    
    @scale.setter
    def scale(self, value):
        """
        Setter for `GammaDistribution.scale`.
        
        Parameters
        ----------
        value : float
            a positive number
        """
        if type(value) in numerical_types:
            if value > 0:
                self._scale = value
            else:
                raise ValueError("scale given to GammaDistribution was not " +\
                    "positive.")
        else:
            raise TypeError("scale given to GammaDistribution was not a " +\
                "number.")
    
    @property
    def power(self):
        """
        The power parameter, \\(p\\), of the `GammaDistribution`. This is the
        power of x in the exponent in the pdf.
        """
        if not hasattr(self, '_power'):
            raise AttributeError("power was referenced before it was set.")
        return self._power
    
    @power.setter
    def power(self, value):
        """
        Setter for `GammaDistribution.power`.
        
        Parameters
        ----------
        value : float
            a positive number
        """
        if type(value) in numerical_types:
            if value > 0:
                self._power = value
            else:
                raise ValueError("power given to GammaDistribution was not " +\
                    "positive.")
        else:
            raise TypeError("power given to GammaDistribution was not a " +\
                "number.")
    
    @property
    def numparams(self):
        """
        The number of parameters of this `GammaDistribution`, 1.
        """
        return 1
    
    @property
    def mean(self):
        """
        The mean of this `GammaDistribution`,
        \\(\\theta\\frac{\\Gamma[(k+1)/p]}{\\Gamma[k/p]}\\).
        """
        if not hasattr(self, '_mean'):
            self._mean = self.scale *\
                np.exp(gammaln((self.shape + 1) / self.power) -\
                gammaln(self.shape / self.power))
        return self._mean
    
    @property
    def variance(self):
        """
        The variance of this `GammaDistribution`,
        \\(\\theta^2 \\left\\{ \\frac{\\Gamma[(k+2)/p]}{\\Gamma[k/p]} -\
        \\left(\\frac{\\Gamma[(k+1)/p]}{\\Gamma[k/p]}\\right)^2 \\right\\}\\).
        """
        if not hasattr(self, '_variance'):
            first_term = np.exp(gammaln((self.shape + 2) / self.power) -\
                gammaln(self.shape / self.power))
            second_term = (np.exp(gammaln((self.shape + 1) / self.power) -\
                gammaln(self.shape / self.power)) ** 2)
            self._variance = (self.scale ** 2) * (first_term - second_term)
        return self._variance
    
    def draw(self, shape=None, random=rand):
        """
        Draws point(s) from this `GammaDistribution`.
        
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
        return self.inverse_cdf(random.uniform(size=shape))
    
    @property
    def log_value_constant(self):
        """
        The constant in the log pdf of this distribution, given by
        \\(\\ln{p}-k\\ln{\\theta}-\\ln{\\Gamma(k/p)}\\).
        """
        if not hasattr(self, '_log_value_constant'):
            self._log_value_constant = np.log(self.power) -\
                (self.shape * np.log(self.scale)) -\
                gammaln(self.shape / self.power)
        return self._log_value_constant
    
    def log_value(self, point):
        """
        Computes the logarithm of the value of this `GammaDistribution` at the
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
        return self.log_value_constant +\
            (self.shape_minus_one * np.log(point)) -\
            np.power(point / self.scale, self.power)
    
    def to_string(self):
        """
        Finds and returns a string version of this `GammaDistribution` of
        the form `"Gamma(k,theta,p)"`.
        """
        return "Gamma({0:.2g}, {1:.2g}, {2:.2g})".format(self.shape,\
            self.scale, self.power)
    
    def __eq__(self, other):
        """
        Checks for equality of this `GammaDistribution` with `other`.
        
        Parameters
        ----------
        other : object
            object to check for equality
        
        Returns
        -------
        result : bool
            True if and only if `other` is a `GammaDistribution` with the same
            `GammaDistribution.shape`, `GammaDistribution.scale`, and
            `GammaDistribution.power`
        """
        if isinstance(other, GammaDistribution):
            tol_kwargs = {'rtol': 1e-9, 'atol': 0}
            shape_equal = np.isclose(self.shape, other.shape, **tol_kwargs)
            scale_equal = np.isclose(self.scale, other.scale, **tol_kwargs)
            power_equal = np.isclose(self.power, other.power, **tol_kwargs)
            metadata_equal = self.metadata_equal(other)
            return all([shape_equal, scale_equal, power_equal, metadata_equal])
        else:
            return False
    
    def inverse_cdf(self, cdf):
        """
        Computes the inverse of the cumulative distribution function (cdf) of
        this `GammaDistribution`.
        
        Parameters
        ----------
        cdf : float
            probability value between 0 and 1
        
        Returns
        -------
        point : float
            value which yields `cdf` when it the CDF is evaluated at it
        """
        return self.scale *\
            np.power(gammaincinv(self.shape / self.power, cdf), 1 / self.power)
    
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
        `GammaDistribution` so that it can be loaded later.
        
        Parameters
        ----------
        group : h5py.Group
            hdf5 file group to fill
        save_metadata : bool
            - if True, attempts to save metadata alongside distribution and
            throws error if it fails
            - if False, metadata is ignored in saving process
        """
        group.attrs['class'] = 'GammaDistribution'
        group.attrs['shape'] = self.shape
        group.attrs['scale'] = self.scale
        group.attrs['power'] = self.power
        if save_metadata:
            self.save_metadata(group)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a `GammaDistribution` from the given hdf5 file group.
        
        Parameters
        ----------
        group : h5py.Group
            the same hdf5 file group which fill_hdf5_group was called on when
            this Distribution was saved
        
        Returns
        -------
        distribution : `GammaDistribution`
            distribution created from the information in the given group
        """
        try:
            assert group.attrs['class'] == 'GammaDistribution'
        except:
            raise TypeError("The given hdf5 file doesn't seem to contain a " +\
                "GammaDistribution.")
        metadata = Distribution.load_metadata(group)
        shape = group.attrs['shape']
        scale = group.attrs['scale']
        power = group.attrs['power']
        return GammaDistribution(shape, scale=scale, power=power,\
            metadata=metadata)
    
    @property
    def gradient_computable(self):
        """
        Boolean describing whether the gradient of the given distribution has
        been implemented. If True,
        `GammaDistribution.gradient_of_log_value` method can be called safely.
        """
        return True
    
    def gradient_of_log_value(self, point):
        """
        Computes the gradient (derivative) of the logarithm of the value of
        this `GammaDistribution` at the given point.
        
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
        return ((self.shape_minus_one / point) -\
            ((self.power / self.scale) *\
            np.power(point / self.scale, self.power - 1)))
    
    @property
    def hessian_computable(self):
        """
        Boolean describing whether the hessian of the given distribution has
        been implemented. If True,
        `GammaDistribution.hessian_of_log_value` method can be called safely.
        """
        return True
    
    def hessian_of_log_value(self, point):
        """
        Computes the hessian (second derivative) of the logarithm of the value
        of this `GammaDistribution` at the given point.
        
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
        return ((-1) * self.shape_minus_one) / np.power(point, 2) -\
            (((self.power * (self.power - 1)) / (self.scale ** 2)) *\
            np.power(point / self.scale, self.power - 2))
    
    def copy(self):
        """
        Copies this distribution.
        
        Returns
        -------
        copied : `GammaDistribution`
            a deep copy of this distribution, ignoring metadata.
        """
        return GammaDistribution(self.shape, self.scale, self.power)

