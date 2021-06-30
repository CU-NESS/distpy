"""
Module containing class representing a distribution based on the hyperbolic
secant function. Its PDF is represented by: $$f(x) = \\frac{1}{2\\sigma}\\ \
\\text{sech}{\\left[\\pi\\left(\\frac{x-\\mu}{2\\sigma}\\right)\\right]}$$

**File**: $DISTPY/distpy/distribution/SechDistribution.py  
**Author**: Keith Tauscher  
**Date**: 31 May 2021
"""
import numpy as np
import numpy.random as rand
from ..util import numerical_types
from .Distribution import Distribution

pi_over_2 = (np.pi / 2)

class SechDistribution(Distribution):
    """
    Class representing a distribution based on the hyperbolic secant function.
    Its PDF is represented by: $$f(x) = \\frac{1}{2\\sigma}\\ \
    \\text{sech}{\\left[\\pi\\left(\\frac{x-\\mu}{2\\sigma}\\right)\\right]}$$
    """
    def __init__(self, mean, variance, metadata=None):
        """
        Initializes a new `SechDistribution` with the given parameter values.
        
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
        The mean of this `SechDistribution`, \\(\\mu\\).
        """
        if not hasattr(self, '_mean'):
            raise AttributeError("mean was referenced before it was set.")
        return self._mean
    
    @mean.setter
    def mean(self, value):
        """
        Setter for `SechDistribution.mean`.
        
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
        The variance of this `SechDistribution`, \\(\\sigma^2\\).
        """
        if not hasattr(self, '_variance'):
            raise AttributeError("variance was referenced before it was set.")
        return self._variance
    
    @variance.setter
    def variance(self, value):
        """
        Setter for `SechDistribution.variance`.
        
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
    def standard_deviation(self):
        """
        The standard deviation of the distribution.
        """
        if not hasattr(self, '_standard_deviation'):
            self._standard_deviation = np.sqrt(self.variance)
        return self._standard_deviation

    @property
    def numparams(self):
        """
        The number of parameters of this `SechDistribution`, 1.
        """
        return 1

    def draw(self, shape=None, random=rand):
        """
        Draws point(s) from this `SechDistribution`.
        
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
        return (self.mean + ((self.standard_deviation / pi_over_2) *\
            np.log(np.tan(pi_over_2 * random.uniform(size=shape)))))

    def log_value(self, point):
        """
        Computes the logarithm of the value of this `SechDistribution` at the
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
        return -np.log(2 * self.standard_deviation * np.cosh((np.pi / 2) *\
            ((point - self.mean) / self.standard_deviation)))
    
    def to_string(self):
        """
        Finds and returns a string version of this `SechDistribution` of the
        form `"Sech(mean, variance)"`.
        """
        return "Sech({0:.2g}, {1:.2g})".format(self.mean, self.variance)
    
    def __eq__(self, other):
        """
        Checks for equality of this `SechDistribution` with `other`.
        
        Parameters
        ----------
        other : object
            object to check for equality
        
        Returns
        -------
        result : bool
            True if and only if `other` is a `SechDistribution` with the same
            `SechDistribution.mean`
        """
        if isinstance(other, SechDistribution):
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
        this `SechDistribution`.
        
        Parameters
        ----------
        cdf : float
            probability value between 0 and 1
        
        Returns
        -------
        point : float
            value which yields `cdf` when it the CDF is evaluated at it
        """
        return (self.mean + ((self.standard_deviation / pi_over_2) *\
            np.log(np.tan(pi_over_2 * cdf))))
    
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
        Fills the given hdf5 file group with data about this `SechDistribution`
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
        group.attrs['class'] = 'SechDistribution'
        group.attrs['mean'] = self.mean
        group.attrs['variance'] = self.variance
        if save_metadata:
            self.save_metadata(group)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a `SechDistribution` from the given hdf5 file group.
        
        Parameters
        ----------
        group : h5py.Group
            the same hdf5 file group which fill_hdf5_group was called on when
            this Distribution was saved
        
        Returns
        -------
        distribution : `SechDistribution`
            distribution created from the information in the given group
        """
        try:
            assert group.attrs['class'] == 'SechDistribution'
        except:
            raise TypeError("The given hdf5 file doesn't seem to contain a " +\
                "SechDistribution.")
        metadata = Distribution.load_metadata(group)
        mean = group.attrs['mean']
        variance = group.attrs['variance']
        return SechDistribution(mean, variance, metadata=metadata)
    
    @property
    def gradient_computable(self):
        """
        Boolean describing whether the gradient of the given distribution has
        been implemented. If True,
        `SechDistribution.gradient_of_log_value` method can be called safely.
        """
        return True
    
    def gradient_of_log_value(self, point):
        """
        Computes the gradient (derivative) of the logarithm of the value of
        this `SechDistribution` at the given point.
        
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
        return (pi_over_2 / (-self.standard_deviation)) * np.tanh(\
            pi_over_2 * ((point - self.mean) / self.standard_deviation))
    
    @property
    def hessian_computable(self):
        """
        Boolean describing whether the hessian of the given distribution has
        been implemented. If True,
        `SechDistribution.hessian_of_log_value` method can be called safely.
        """
        return True
    
    def hessian_of_log_value(self, point):
        """
        Computes the hessian (second derivative) of the logarithm of the value
        of this `SechDistribution` at the given point.
        
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
        return -np.power((self.standard_deviation / pi_over_2) * np.cosh(\
            pi_over_2 * ((point - self.mean) / self.standard_deviation)), -2)
    
    def copy(self):
        """
        Copies this distribution.
        
        Returns
        -------
        copied : `SechDistribution`
            a deep copy of this distribution, ignoring metadata.
        """
        return SechDistribution(self.mean, self.variance)

