"""
Module containing class representing a Weibull distribution. Its PDF is
represented by: $$f(x) = \\frac{k}{\\theta}\
\\left(\\frac{x}{\\theta}\\right)^{k-1}\\ e^{-(x/\\theta)^k}$$

**File**: $DISTPY/distpy/distribution/WeibullDistribution.py  
**Author**: Keith Tauscher  
**Date**: 31 May 2021
"""
from __future__ import division
import numpy as np
import numpy.random as rand
from scipy.special import gamma
from ..util import numerical_types
from .Distribution import Distribution

class WeibullDistribution(Distribution):
    """
    Class representing a Weibull distribution. Its PDF is represented by:
    $$f(x) = \\frac{k}{\\theta}\\left(\\frac{x}{\\theta}\\right)^{k-1}\\ \
    e^{-(x/\\theta)^k}$$
    """
    def __init__(self, shape=1, scale=1., metadata=None):
        """
        Initializes a new `WeibullDistribution` with the given parameter
        values.
        
        Parameters
        ----------
        shape : float
            positive real number, \\(k\\)
        scale : float
            positive real number, \\(\\theta\\)
        metadata : number or str or dict or `distpy.util.Savable.Savable`
            data to store alongside this distribution.
        """
        self.shape = shape
        self.scale = scale
        self.metadata = metadata
    
    @property
    def shape(self):
        """
        The shape parameter, \\(k\\), of the distribution.
        """
        if not hasattr(self, '_shape'):
            raise AttributeError("shape was referenced before it was set.")
        return self._shape
    
    @shape.setter
    def shape(self, value):
        """
        Setter for `WeibullDistribution.shape`.
        
        Parameters
        ----------
        value : float
            a positive number
        """
        if type(value) in numerical_types:
            if value > 0:
                self._shape = value
            else:
                raise ValueError("shape parameter of WeibullDistribution " +\
                    "was not positive.")
        else:
            raise TypeError("shape parameter of WeibullDistribution was " +\
                "not a number.")
    
    @property
    def scale(self):
        """
        The scale parameter, \\(\\theta\\), of the distribution.
        """
        if not hasattr(self, '_scale'):
            raise AttributeError("scale was referenced before it was set.")
        return self._scale
    
    @scale.setter
    def scale(self, value):
        """
        Setter for `WeibullDistribution.scale`.
        
        Parameters
        ----------
        value : float
            a positive number
        """
        if type(value) in numerical_types:
            if value > 0:
                self._scale = value
            else:
                raise ValueError("scale parameter of WeibullDistribution " +\
                    "was not positive.")
        else:
            raise TypeError("scale parameter of WeibullDistribution was " +\
                "not a number.")
    
    @property
    def const_lp_term(self):
        """
        The constant part of the log probability density.
        """
        if not hasattr(self, '_const_lp_term'):
            self._const_lp_term =\
                np.log(self.shape) - (self.shape * np.log(self.scale))
        return self._const_lp_term

    @property
    def numparams(self):
        """
        The number of parameters of this `WeibullDistribution`, 1.
        """
        return 1
    
    @property
    def mean(self):
        """
        The mean of this `WeibullDistribution`,
        \\(\\theta\\Gamma\\left(1+\\frac{1}{k}\\right)\\).
        """
        if not hasattr(self, '_mean'):
            self._mean = self.scale * gamma(1 + (1 / self.shape))
        return self._mean
    
    @property
    def variance(self):
        """
        The variance of this `WeibullDistribution`,
        \\(\\theta^2\\left\\{\\Gamma\\left(1+\\frac{2}{k}\\right)-\
        \\left[\\Gamma\\left(1+\\frac{1}{k}\\right)\\right]^2\\right\\}\\).
        """
        if not hasattr(self, '_variance'):
            self._variance = (self.scale ** 2) *\
                (gamma(1 + (2 / self.shape)) -\
                (gamma(1 + (1 / self.shape)) ** 2))
        return self._variance

    def draw(self, shape=None, random=rand):
        """
        Draws point(s) from this `WeibullDistribution`.
        
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
        return self.scale * random.weibull(self.shape, size=shape)


    def log_value(self, point):
        """
        Computes the logarithm of the value of this `WeibullDistribution` at
        the given point.
        
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
        if point >= 0:
            return self.const_lp_term + ((self.shape - 1) * np.log(point)) -\
                np.power(point / self.scale, self.shape)
        return -np.inf
    
    def to_string(self):
        """
        Finds and returns a string version of this `WeibullDistribution` of the
        form `"Weibull(k,theta)"`.
        """
        return "Weibull({0:.2g}, {1:.2g})".format(self.shape, self.scale)
    
    def __eq__(self, other):
        """
        Checks for equality of this `WeibullDistribution` with `other`.
        
        Parameters
        ----------
        other : object
            object to check for equality
        
        Returns
        -------
        result : bool
            True if and only if `other` is a `WeibullDistribution` with the
            same `WeibullDistribution.shape` and `WeibullDistribution.scale`
        """
        if isinstance(other, WeibullDistribution):
            tol_kwargs = {'rtol': 0., 'atol': 1e-9}
            shape_close = np.isclose(self.shape, other.shape, **tol_kwargs)
            scale_close = np.isclose(self.scale, other.scale, **tol_kwargs)
            metadata_equal = self.metadata_equal(other)
            return all([shape_close, scale_close, metadata_equal])
        else:
            return False
    
    def inverse_cdf(self, cdf):
        """
        Computes the inverse of the cumulative distribution function (cdf) of
        this `WeibullDistribution`.
        
        Parameters
        ----------
        cdf : float
            probability value between 0 and 1
        
        Returns
        -------
        point : float
            value which yields `cdf` when it the CDF is evaluated at it
        """
        return (self.scale * np.power(-np.log(1 - cdf), 1 / self.shape))
    
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
        `WeibullDistribution` so that it can be loaded later.
        
        Parameters
        ----------
        group : h5py.Group
            hdf5 file group to fill
        save_metadata : bool
            - if True, attempts to save metadata alongside distribution and
            throws error if it fails
            - if False, metadata is ignored in saving process
        """
        group.attrs['class'] = 'WeibullDistribution'
        group.attrs['shape'] = self.shape
        group.attrs['scale'] = self.scale
        if save_metadata:
            self.save_metadata(group)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a `WeibullDistribution` from the given hdf5 file group.
        
        Parameters
        ----------
        group : h5py.Group
            the same hdf5 file group which fill_hdf5_group was called on when
            this Distribution was saved
        
        Returns
        -------
        distribution : `WeibullDistribution`
            distribution created from the information in the given group
        """
        try:
            assert group.attrs['class'] == 'WeibullDistribution'
        except:
            raise TypeError("The given hdf5 file doesn't seem to contain a " +\
                "WeibullDistribution.")
        metadata = Distribution.load_metadata(group)
        shape = group.attrs['shape']
        scale = group.attrs['scale']
        return WeibullDistribution(shape=shape, scale=scale, metadata=metadata)
    
    @property
    def gradient_computable(self):
        """
        Boolean describing whether the gradient of the given distribution has
        been implemented. If True,
        `WeibullDistribution.gradient_of_log_value` method can be called
        safely.
        """
        return True
    
    def gradient_of_log_value(self, point):
        """
        Computes the gradient (derivative) of the logarithm of the value of
        this `WeibullDistribution` at the given point.
        
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
        return (((self.shape - 1) / point) - ((self.shape / self.scale) *\
            ((point / self.scale) ** (self.shape - 1))))
    
    @property
    def hessian_computable(self):
        """
        Boolean describing whether the hessian of the given distribution has
        been implemented. If True,
        `WeibullDistribution.hessian_of_log_value` method can be called
        safely.
        """
        return True
    
    def hessian_of_log_value(self, point):
        """
        Computes the hessian (second derivative) of the logarithm of the value
        of this `WeibullDistribution` at the given point.
        
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
        return (((1 - self.shape) / (point ** 2)) -\
            ((self.shape / self.scale) * ((self.shape - 1) / self.scale) *\
            ((point / self.scale) ** (self.shape - 2))))
    
    def copy(self):
        """
        Copies this distribution.
        
        Returns
        -------
        copied : `WeibullDistribution`
            a deep copy of this distribution, ignoring metadata.
        """
        return WeibullDistribution(self.shape, self.scale)

