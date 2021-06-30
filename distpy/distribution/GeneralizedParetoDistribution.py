"""
Module containing class representing a generalized Pareto distribution. Its PDF
is represented by: $$f(x) = \\left(\\frac{\\alpha-1}{\\sigma}\\right)\\ \
\\left[1 + \\left(\\frac{x-\\mu}{\\sigma}\\right)\\right]^{\\alpha},$$ where
\\(x\\ge\\mu\\).

**File**: $DISTPY/distpy/distribution/ABCDEFDistribution.py  
**Author**: Keith Tauscher  
**Date**: 31 May 2021
"""
import numpy as np
import numpy.random as rand
from ..util import int_types, numerical_types
from .Distribution import Distribution

class GeneralizedParetoDistribution(Distribution):
    """
    Class representing a generalized Pareto distribution. Its PDF is
    represented by: $$f(x) = \\left(\\frac{\\alpha-1}{\\sigma}\\right)\\ \
    \\left[1 + \\left(\\frac{x-\\mu}{\\sigma}\\right)\\right]^{\\alpha},$$
    where \\(x\\ge\\mu\\).
    """
    def __init__(self, shape, location=0, scale=1, metadata=None):
        """
        Initializes a new `GeneralizedParetoDistribution` with the given
        parameter values.
        
        Parameters
        ----------
        shape : float
            positive real number, \\(\\alpha\\)
        location : float
            real number, \\(\\mu\\)
        scale : float
            positive real number \\(\\sigma\\)
        metadata : number or str or dict or `distpy.util.Savable.Savable`
            data to store alongside this distribution.
        """
        self.shape = shape
        self.location = location
        self.scale = scale
        self.metadata = metadata
    
    @property
    def shape(self):
        """
        The shape parameter, \\(\\alpha\\), of this distribution.
        """
        if not hasattr(self, '_shape'):
            raise AttributeError("shape was referenced before it was set.")
        return self._shape
    
    @shape.setter
    def shape(self, value):
        """
        Setter for `GeneralizedParetoDistribution.shape`.
        
        Parameters
        ----------
        value : float
            number greater than 1
        """
        if type(value) in numerical_types:
            if value > 1:
                self._shape = float(value)
            else:
                raise ValueError("shape was set to a number less than 1.")
        else:
            raise TypeError("shape parameter was set to a non-number.")
    
    @property
    def location(self):
        """
        Property storing the location parameter, \\(\\mu\\), of this
        distribution.
        """
        if not hasattr(self, '_location'):
            raise AttributeError("location was referenced before it was set.")
        return self._location
    
    @location.setter
    def location(self, value):
        """
        Setter for the `GeneralizedParetoDistribution.location`
        
        Parameters
        ----------
        value : float
            real number
        """
        if type(value) in numerical_types:
            self._location = float(value)
        else:
            raise TypeError("location parameter was set to a non-number.")
    
    @property
    def scale(self):
        """
        The scale parameter, \\(\\sigma\\), of this distribution.
        """
        if not hasattr(self, '_scale'):
            raise AttributeError("scale was referenced before it was set.")
        return self._scale
    
    @scale.setter
    def scale(self, value):
        """
        Setter for `GeneralizedParetoDistribution.scale`.
        
        Parameters
        ----------
        value : float
            positive real number
        """
        if type(value) in numerical_types:
            if value > 0:
                self._scale = float(value)
            else:
                raise ValueError("scale was set to a number less than 0.")
        else:
            raise TypeError("scale parameter was set to a non-number.")
    
    @property
    def numparams(self):
        """
        The number of parameters of this `GeneralizedParetoDistribution`, 1.
        """
        return 1
    
    @property
    def mean(self):
        """
        The mean of this `GeneralizedParetoDistribution`,
        \\(\\mu+\\frac{\\sigma}{\\alpha-2}\\).
        """
        if not hasattr(self, '_mean'):
            if self.shape <= 2:
                raise NotImplementedError("mean is not defined because " +\
                    "shape <= 2.")
            else:
                self._mean = self.location + (self.scale / (self.shape - 2))
        return self._mean
    
    @property
    def variance(self):
        """
        The variance of this `GeneralizedParetoDistribution`,
        \\(\\frac{\\sigma^2(\\alpha-1)}{(\\alpha-3)(\\alpha-2)^2}\\).
        """
        if not hasattr(self, '_variance'):
            if self.shape <= 3:
                raise NotImplementedError("variance is not defined because " +\
                    "shape <= 3.")
            else:
                self._variance = ((self.scale ** 2) * (self.shape - 1)) /\
                    ((self.shape - 3) * ((self.shape - 2) ** 2))
        return self._variance
    
    def draw(self, shape=None, random=rand):
        """
        Draws point(s) from this `GeneralizedParetoDistribution`.
        
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
        if type(shape) is type(None):
            shape = ()
        if type(shape) in int_types:
            shape = (shape,)
        uniforms = random.rand(*shape)
        return self.location +\
            (self.scale * (np.power(uniforms, 1 / (1 - self.shape)) - 1))
    
    @property
    def log_value_constant(self):
        """
        The constant part of the log of the distribution, given by
        \\(\\ln{\\frac{\\alpha-1}{\\theta}}\\).
        """
        if not hasattr(self, '_log_value_constant'):
            self._log_value_constant = np.log((self.shape - 1) / self.scale)
        return self._log_value_constant
    
    def log_value(self, point):
        """
        Computes the logarithm of the value of this
        `GeneralizedParetoDistribution` at the given point.
        
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
        if point >= self.location:
            return self.log_value_constant - (self.shape *\
                np.log(1 + ((point - self.location) / self.scale)))
        return -np.inf
    
    def to_string(self):
        """
        Finds and returns a string version of this
        `GeneralizedParetoDistribution` of the form
        `"Pareto(alpha, mu, sigma)"`.
        """
        return "Pareto({0:.2g}, {1:.2g}, {2:.2g})".format(self.shape,\
            self.location, self.scale)
    
    def __eq__(self, other):
        """
        Checks for equality of this `GeneralizedParetoDistribution` with
        `other`.
        
        Parameters
        ----------
        other : object
            object to check for equality
        
        Returns
        -------
        result : bool
            True if and only if `other` is a `GeneralizedParetoDistribution`
            with the same `GeneralizedParetoDistribution.shape`,
            `GeneralizedParetoDistribution.location`, and
            `GeneralizedParetoDistribution.scale`
        """
        if isinstance(other, GeneralizedParetoDistribution):
            shape_close =\
                np.isclose(self.shape, other.shape, atol=1e-9, rtol=0)
            location_close =\
                np.isclose(self.location, other.location, atol=1e-9, rtol=0)
            scale_close =\
                np.isclose(self.scale, other.scale, atol=0, rtol=1e-9)
            metadata_equal = self.metadata_equal(other)
            return\
                all([shape_close, location_close, scale_close, metadata_equal])
        else:
            return False
    
    def inverse_cdf(self, cdf):
        """
        Computes the inverse of the cumulative distribution function (cdf) of
        this `GeneralizedParetoDistribution`.
        
        Parameters
        ----------
        cdf : float
            probability value between 0 and 1
        
        Returns
        -------
        point : float
            value which yields `cdf` when it the CDF is evaluated at it
        """
        return self.location +\
            (self.scale * (np.power(cdf, 1 / (1 - self.shape)) - 1))
    
    @property
    def minimum(self):
        """
        The minimum allowable value(s) in this distribution.
        """
        return self.location
    
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
        `GeneralizedParetoDistribution` so that it can be loaded later.
        
        Parameters
        ----------
        group : h5py.Group
            hdf5 file group to fill
        save_metadata : bool
            - if True, attempts to save metadata alongside distribution and
            throws error if it fails
            - if False, metadata is ignored in saving process
        """
        group.attrs['class'] = 'GeneralizedParetoDistribution'
        group.attrs['shape'] = self.shape
        group.attrs['location'] = self.location
        group.attrs['scale'] = self.scale
        if save_metadata:
            self.save_metadata(group)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a `GeneralizedParetoDistribution` from the given hdf5 file group.
        
        Parameters
        ----------
        group : h5py.Group
            the same hdf5 file group which fill_hdf5_group was called on when
            this Distribution was saved
        
        Returns
        -------
        distribution : `GeneralizedParetoDistribution`
            distribution created from the information in the given group
        """
        try:
            assert group.attrs['class'] == 'GeneralizedParetoDistribution'
        except:
            raise TypeError("The given hdf5 file doesn't seem to contain a " +\
                "GeneralizedParetoDistribution.")
        metadata = Distribution.load_metadata(group)
        shape = group.attrs['shape']
        location = group.attrs['location']
        scale = group.attrs['scale']
        return GeneralizedParetoDistribution(shape, location=location,\
            scale=scale, metadata=metadata)
    
    @property
    def gradient_computable(self):
        """
        Boolean describing whether the gradient of the given distribution has
        been implemented. If True,
        `GeneralizedParetoDistribution.gradient_of_log_value` method can be
        called safely.
        """
        return True
    
    def gradient_of_log_value(self, point):
        """
        Computes the gradient (derivative) of the logarithm of the value of
        this `GeneralizedParetoDistribution` at the given point.
        
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
        return self.shape / ((self.location - point) - self.scale)
    
    @property
    def hessian_computable(self):
        """
        Boolean describing whether the hessian of the given distribution has
        been implemented. If True,
        `GeneralizedParetoDistribution.hessian_of_log_value` method can be
        called safely.
        """
        return True
    
    def hessian_of_log_value(self, point):
        """
        Computes the hessian (second derivative) of the logarithm of the value
        of this `GeneralizedParetoDistribution` at the given point.
        
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
        return self.shape / (((self.location - point) - self.scale) ** 2)
    
    def copy(self):
        """
        Copies this distribution.
        
        Returns
        -------
        copied : `GeneralizedParetoDistribution`
            a deep copy of this distribution, ignoring metadata.
        """
        return GeneralizedParetoDistribution(self.shape, self.location,\
            self.scale)

