"""
Module containing class representing a double-sided exponential distribution.
Its PDF is represented by: $$f(x) = \\frac{e^{-|x-\\mu|/\\theta}}{2\\theta}$$

**File**: $DISTPY/distpy/distribution/ABCDEFDistribution.py  
**Author**: Keith Tauscher  
**Date**: 31 May 2021
"""
from __future__ import division
import numpy as np
import numpy.random as rand
from ..util import bool_types, numerical_types
from .Distribution import Distribution

class DoubleSidedExponentialDistribution(Distribution):
    """
    Class representing a double-sided exponential distribution. Its PDF is
    represented by: $$f(x) = \\frac{e^{-|x-\\mu|/\\theta}}{2\\theta}$$
    """
    def __init__(self, mean, variance, metadata=None):
        """
        Initializes a new `DoubleSidedExponentialDistribution` with the given
        parameter values.
        
        Parameters
        ----------
        mean : float
            real number, \\(\\mu\\)
        variance : float
            positive number, equal to \\(2\\theta^2\\)
        metadata : number or str or dict or `distpy.util.Savable.Savable`
            data to store alongside this distribution.
        """
        self.mean = mean
        self.variance = variance
        self.metadata = metadata
    
    @property
    def mean(self):
        """
        The mean of this `DoubleSidedExponentialDistribution`, \\(\\mu\\).
        """
        if not hasattr(self, '_mean'):
            raise AttributeError("mean was referenced before it was set.")
        return self._mean
    
    @mean.setter
    def mean(self, value):
        """
        Setter for `DoubleSidedExponentialDistribution.mean`.
        
        Parameters
        ----------
        value : float
            any real number.
        """
        if type(value) in numerical_types:
            self._mean = (value * 1.)
        else:
            raise ValueError("The mean parameter given to a " +\
                "DoubleSidedExponentialDistribution was not of a numerical " +\
                "type.")
    
    @property
    def variance(self):
        """
        The variance of this `DoubleSidedExponentialDistribution`,
        \\(2\\theta^2\\).
        """
        if not hasattr(self, '_variance'):
            raise AttributeError("variance was referenced before it was set.")
        return self._variance
    
    @variance.setter
    def variance(self, value):
        """
        Setter for `DoubleSidedExponentialDistribution.variance`.
        
        Parameters
        ----------
        value : float
            any positive number.
        """
        if type(value) in numerical_types:
            if value > 0:
                self._variance = (value * 1.)
            else:
                raise ValueError("The variance given to a " +\
                    "DoubleSidedExponentialDistribution was not positive.")
        else:
            raise ValueError("The variance parameter given to a " +\
                "DoubleSidedExponentialDistribution was not of a numerical " +\
                "type.")
    
    @property
    def const_lp_term(self):
        """
        The constant part of the log probability density, given by
        \\(-\\ln{(2\\theta)}\\).
        """
        if not hasattr(self, '_const_lp_term'):
            self._const_lp_term = (np.log(2) + np.log(self.variance)) / (-2)
        return self._const_lp_term
    
    @property
    def numparams(self):
        """
        The number of parameters of this `DoubleSidedExponentialDistribution`,
        1.
        """
        return 1
    
    @property
    def root_half_variance(self):
        """
        The square root of half the variance, \\(\\theta\\).
        """
        if not hasattr(self, '_root_half_variance'):
            self._root_half_variance = np.sqrt(self.variance / 2.)
        return self._root_half_variance
    
    def draw(self, shape=None, random=rand):
        """
        Draws point(s) from this `DoubleSidedExponentialDistribution`.
        
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
        return rand.laplace(loc=self.mean, scale=self.root_half_variance,\
            size=shape)
    
    def log_value(self, point):
        """
        Computes the logarithm of the value of this
        `DoubleSidedExponentialDistribution` at the given point.
        
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
        return self.const_lp_term -\
            (np.abs(point - self.mean) / self.root_half_variance)

    def to_string(self):
        """
        Finds and returns a string version of this
        `DoubleSidedExponentialDistribution` of the form `"DSExp(mean, var)"`.
        """
        return "DSExp({0:.2g}, {1:.2g})".format(self.mean, self.variance)
    
    def __eq__(self, other):
        """
        Checks for equality of this `DoubleSidedExponentialDistribution` with
        `other`.
        
        Parameters
        ----------
        other : object
            object to check for equality
        
        Returns
        -------
        result : bool
            True if and only if `other` is a
            `DoubleSidedExponentialDistribution` with the same
            `DoubleSidedExponentialDistribution.mean` and
            `DoubleSidedExponentialDistribution.variance`
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
        Computes the inverse of the cumulative distribution function (cdf) of
        this `DoubleSidedExponentialDistribution`.
        
        Parameters
        ----------
        cdf : float
            probability value between 0 and 1
        
        Returns
        -------
        point : float
            value which yields `cdf` when it the CDF is evaluated at it
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
        `DoubleSidedExponentialDistribution` so that it can be loaded later.
        
        Parameters
        ----------
        group : h5py.Group
            hdf5 file group to fill
        save_metadata : bool
            - if True, attempts to save metadata alongside distribution and
            throws error if it fails
            - if False, metadata is ignored in saving process
        """
        group.attrs['class'] = 'DoubleSidedExponentialDistribution'
        group.attrs['mean'] = self.mean
        group.attrs['variance'] = self.variance
        if save_metadata:
            self.save_metadata(group)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a `DoubleSidedExponentialDistribution` from the given hdf5 file
        group.
        
        Parameters
        ----------
        group : h5py.Group
            the same hdf5 file group which fill_hdf5_group was called on when
            this Distribution was saved
        
        Returns
        -------
        distribution : `DoubleSidedExponentialDistribution`
            distribution created from the information in the given group
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
        Boolean describing whether the gradient of the given distribution has
        been implemented. If True,
        `DoubleSidedExponentialDistribution.gradient_of_log_value` method can
        be called safely.
        """
        return True
    
    def gradient_of_log_value(self, point):
        """
        Computes the gradient (derivative) of the logarithm of the value of
        this `DoubleSidedExponentialDistribution` at the given point.
        
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
        return np.sign(self.mean - point) / self.root_half_variance
    
    @property
    def hessian_computable(self):
        """
        Boolean describing whether the hessian of the given distribution has
        been implemented. If True,
        `DoubleSidedExponentialDistribution.hessian_of_log_value` method can
        be called safely.
        """
        return True
    
    def hessian_of_log_value(self, point):
        """
        Computes the hessian (second derivative) of the logarithm of the value
        of this `DoubleSidedExponentialDistribution` at the given point.
        
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
        return 0
    
    def copy(self):
        """
        Copies this distribution.
        
        Returns
        -------
        copied : `DoubleSidedExponentialDistribution`
            a deep copy of this distribution, ignoring metadata.
        """
        return DoubleSidedExponentialDistribution(self.mean, self.variance)

