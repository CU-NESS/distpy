"""
Module containing class representing a distribution based on the squared
hyperbolic secant function. Its PDF is represented by: $$f(x) =\
\\frac{\\pi}{4\\sqrt{3}\\ \\sigma}\\ \\text{sech}^2{\\left[\
\\frac{\\pi}{2\\sqrt{3}}\\left(\\frac{x-\\mu}{\\sigma}\\right)\\right]}$$

**File**: $DISTPY/distpy/distribution/ABCDEFDistribution.py  
**Author**: Keith Tauscher  
**Date**: 31 May 2021
"""
import numpy as np
import numpy.random as rand
from ..util import numerical_types
from .Distribution import Distribution

class SechSquaredDistribution(Distribution):
    """
    Class representing a distribution based on the squared hyperbolic secant
    function. Its PDF is represented by:
    $$f(x) = \\frac{\\pi}{4\\sqrt{3}\\ \\sigma}\\ \\text{sech}^2{\\left[\
    \\frac{\\pi}{2\\sqrt{3}}\\left(\\frac{x-\\mu}{\\sigma}\\right)\\right]}$$
    """
    def __init__(self, mean, variance, metadata=None):
        """
        Initializes a new `SechSquaredDistribution` with the given parameter
        values.
        
        Parameters
        ----------
        mean : float
            any real number, \\(\\mu\\)
        variance : float
            any positive real number, \\(\\sigma^2\\)
        metadata : number or str or dict or `distpy.util.Savable.Savable`
            data to store alongside this distribution.
        """
        self.mean = mean
        self.variance = variance
        self.metadata = metadata
    
    @property
    def mean(self):
        """
        The mean of this `SechSquaredDistribution`, \\(\\mu\\).
        """
        if not hasattr(self, '_mean'):
            raise AttributeError("mean was referenced before it was set.")
        return self._mean
    
    @mean.setter
    def mean(self, value):
        """
        Setter for `SechSquaredDistribution.mean`.
        
        Parameters
        ----------
        value : float
            real number center of the distribution
        """
        if type(value) in numerical_types:
            self._mean = value
        else:
            raise TypeError("mean was set to a non-number.")
    
    @property
    def variance(self):
        """
        The variance of this `SechSquaredDistribution`, \\(\\sigma^2\\).
        """
        if not hasattr(self, '_variance'):
            raise AttributeError("variance was referenced before it was set.")
        return self._variance
    
    @variance.setter
    def variance(self, value):
        """
        Setter for `SechSquaredDistribution.variance`.
        
        Parameters
        ----------
        value : float
            positive number squared width of the distribution
        """
        if type(value) in numerical_types:
            if value > 0:
                self._variance = value
            else:
                raise ValueError("variance was set to a non-positive number.")
        else:
            raise TypeError("variance was set to a non-number.")
    
    @property
    def scale(self):
        """
        A measure of the scale of the distribution.
        """
        if not hasattr(self, '_scale'):
            self._scale = 2 * np.sqrt(3 * self.variance) / np.pi
        return self._scale
    
    @property
    def log_value_constant(self):
        """
        The constant part of the log value of this distribution (this is
        related to the log of the normalization constant in the pdf).
        """
        if not hasattr(self, '_log_value_constant'):
            self._log_value_constant = ((-1) * np.log(2 * self.scale))
        return self._log_value_constant
    
    @property
    def numparams(self):
        """
        The number of parameters of this `SechSquaredDistribution`, 1.
        """
        return 1
    
    def draw(self, shape=None, random=rand):
        """
        Draws point(s) from this `SechSquaredDistribution`.
        
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
        return self.mean +\
            (self.scale * np.arctanh((2 * random.uniform(size=shape)) - 1))

    def log_value(self, point):
        """
        Computes the logarithm of the value of this `SechSquaredDistribution`
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
        return self.log_value_constant -\
            (2 * np.log(np.cosh((point - self.mean) / self.scale)))
    
    def to_string(self):
        """
        Finds and returns a string version of this `SechSquaredDistribution` of
        the form `"Sech2(mean, variance)"`.
        """
        return "Sech2({0:.2g}, {1:.2g})".format(self.mean, self.variance)
    
    def __eq__(self, other):
        """
        Checks for equality of this `SechSquaredDistribution` with `other`.
        
        Parameters
        ----------
        other : object
            object to check for equality
        
        Returns
        -------
        result : bool
            True if and only if `other` is a `SechSquaredDistribution` with the
            same `SechSquaredDistribution.mean` and
            `SechSquaredDistribution.variance`
        """
        if isinstance(other, SechSquaredDistribution):
            tol_kwargs = {'rtol': 1e-9, 'atol': 0}
            mean_equal = np.isclose(self.mean, other.mean, **tol_kwargs)
            variance_equal =\
                np.isclose(self.variance, other.variance, **tol_kwargs)
            metadata_equal = self.metadata_equal(other)
            return all([mean_equal, variance_equal, metadata_equal])
        else:
            return False
    
    def inverse_cdf(self, cdf):
        """
        Computes the inverse of the cumulative distribution function (cdf) of
        this `SechSquaredDistribution`.
        
        Parameters
        ----------
        cdf : float
            probability value between 0 and 1
        
        Returns
        -------
        point : float
            value which yields `cdf` when it the CDF is evaluated at it
        """
        return self.mean + (self.scale * np.arctanh((2 * cdf) - 1))
    
    @property
    def minimum(self):
        """
        The minimum allowable value(s) in this distribution.
        """
        return None
    
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
        `SechSquaredDistribution` so that it can be loaded later.
        
        Parameters
        ----------
        group : h5py.Group
            hdf5 file group to fill
        save_metadata : bool
            - if True, attempts to save metadata alongside distribution and
            throws error if it fails
            - if False, metadata is ignored in saving process
        """
        group.attrs['class'] = 'SechSquaredDistribution'
        group.attrs['mean'] = self.mean
        group.attrs['variance'] = self.variance
        if save_metadata:
            self.save_metadata(group)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a `SechSquaredDistribution` from the given hdf5 file group.
        
        Parameters
        ----------
        group : h5py.Group
            the same hdf5 file group which fill_hdf5_group was called on when
            this Distribution was saved
        
        Returns
        -------
        distribution : `SechSquaredDistribution`
            distribution created from the information in the given group
        """
        try:
            assert group.attrs['class'] == 'SechSquaredDistribution'
        except:
            raise TypeError("The given hdf5 file doesn't seem to contain a " +\
                "SechSquaredDistribution.")
        metadata = Distribution.load_metadata(group)
        mean = group.attrs['mean']
        variance = group.attrs['variance']
        return SechSquaredDistribution(mean, variance, metadata=metadata)
    
    @property
    def gradient_computable(self):
        """
        Boolean describing whether the gradient of the given distribution has
        been implemented. If True,
        `SechSquaredDistribution.gradient_of_log_value` method can be called
        safely.
        """
        return True
    
    def gradient_of_log_value(self, point):
        """
        Computes the gradient (derivative) of the logarithm of the value of
        this `SechSquaredDistribution` at the given point.
        
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
        return ((-2) / self.scale) * np.tanh((point - self.mean) / self.scale)
    
    @property
    def hessian_computable(self):
        """
        Boolean describing whether the hessian of the given distribution has
        been implemented. If True,
        `SechSquaredDistribution.hessian_of_log_value` method can be called
        safely.
        """
        return True
    
    def hessian_of_log_value(self, point):
        """
        Computes the hessian (second derivative) of the logarithm of the value
        of this `SechSquaredDistribution` at the given point.
        
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
        return (((-2) * (np.pi ** 2)) / (3 * self.variance)) *\
            np.power((point - self.mean) / self.scale, -2)
    
    def copy(self):
        """
        Copies this distribution.
        
        Returns
        -------
        copied : `SechSquaredDistribution`
            a deep copy of this distribution, ignoring metadata.
        """
        return SechSquaredDistribution(self.mean, self.variance)

