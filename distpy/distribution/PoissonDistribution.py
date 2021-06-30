"""
Module containing class representing a Poisson distribution. Its PMF is
represented by: $$f(x) = \\frac{\\lambda^x}{x!}\\ e^{-\\lambda},$$ where
\\(x\\) is a non-negative integer.

**File**: $DISTPY/distpy/distribution/PoissonDistribution.py  
**Author**: Keith Tauscher  
**Date**: 31 May 2021
"""
import numpy as np
import numpy.random as rand
from scipy.special import gammaln as log_gamma
from ..util import int_types, numerical_types
from .Distribution import Distribution

class PoissonDistribution(Distribution):
    """
    Class representing a Poisson distribution. Its PMF is represented by:
    $$f(x) = \\frac{\\lambda^x}{x!}\\ e^{-\\lambda},$$ where \\(x\\) is a
    non-negative integer.
    """
    def __init__(self, scale, metadata=None):
        """
        Initializes a new `PoissonDistribution` with the given parameter
        values.
        
        Parameters
        ----------
        scale : float
            the mean and variance, \\(\\lambda\\) of the distribution
        metadata : number or str or dict or `distpy.util.Savable.Savable`
            data to store alongside this distribution.
        """
        self.scale = scale
        self.metadata = metadata
    
    @property
    def scale(self):
        """
        The scale parameter, \\(\\lambda\\), of this Poisson distribution.
        """
        if not hasattr(self, '_scale'):
            raise AttributeError("scale was referenced before it was set.")
        return self._scale
    
    @scale.setter
    def scale(self, value):
        """
        Setter for `PoissonDistribution.scale`.
        
        Parameters
        ----------
        value : float
            positive number that is both mean and variance of this distribution
        """
        if type(value) in numerical_types:
            if value > 0:
                self._scale = (value * 1.)
            else:
                raise ValueError("scale given to PoissonDistribution was " +\
                    "not positive.")
        else:
            raise ValueError("scale given to PoissonDistribution was not a " +\
                "number.")
    
    @property
    def numparams(self):
        """
        The number of parameters of this `PoissonDistribution`, 1.
        """
        return 1
    
    @property
    def mean(self):
        """
        The mean of this `PoissonDistribution`, \\(\\lambda\\).
        """
        if not hasattr(self, '_mean'):
            self._mean = self.scale
        return self._mean
    
    @property
    def variance(self):
        """
        The variance of this `BernoulliDistribution`, \\(\\lambda\\).
        """
        if not hasattr(self, '_variance'):
            self._variance = self.scale
        return self._variance
    
    def draw(self, shape=None, random=rand):
        """
        Draws point(s) from this `PoissonDistribution`.
        
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
        return random.poisson(lam=self.scale, size=shape)
    
    def log_value(self, point):
        """
        Computes the logarithm of the value of this `PoissonDistribution` at
        the given point.
        
        Parameters
        ----------
        point : int
            scalar at which to evaluate PDF
        
        Returns
        -------
        value : float
            natural logarithm of the value of this distribution at `point`. If
            \\(f\\) is this distribution's PDF and \\(x\\) is `point`, then
            `value` is \\(\\ln{\\big(f(x)\\big)}\\)
        """
        if type(point) in int_types:
            if point >= 0:
                return (point * np.log(self.scale)) - self.scale -\
                    log_gamma(point + 1)
            else:
                return -np.inf
        else:
            raise TypeError("point given to PoissonDistribution was not an " +\
                "integer.")

    def to_string(self):
        """
        Finds and returns a string version of this `PoissonDistribution` of the
        form `"Poisson(lambda)"`.
        """
        return "Poisson({:.4g})".format(self.scale)
    
    def __eq__(self, other):
        """
        Checks for equality of this `PoissonDistribution` with `other`.
        
        Parameters
        ----------
        other : object
            object to check for equality
        
        Returns
        -------
        result : bool
            True if and only if `other` is a `PoissonDistribution` with the
            same `PoissonDistribution.scale`
        """
        if isinstance(other, PoissonDistribution):
            scale_close =\
                np.isclose(self.scale, other.scale, rtol=1e-6, atol=1e-6)
            metadata_equal = self.metadata_equal(other)
            return (scale_close and metadata_equal)
        else:
            return False
    
    @property
    def can_give_confidence_intervals(self):
        """
        Discrete distributions do not support confidence intervals.
        """
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
        return None
    
    @property
    def is_discrete(self):
        """
        Boolean describing whether this distribution is discrete (True) or
        continuous (False).
        """
        return True
    
    def fill_hdf5_group(self, group, save_metadata=True):
        """
        Fills the given hdf5 file group with data about this
        `PoissonDistribution` so that it can be loaded later.
        
        Parameters
        ----------
        group : h5py.Group
            hdf5 file group to fill
        save_metadata : bool
            - if True, attempts to save metadata alongside distribution and
            throws error if it fails
            - if False, metadata is ignored in saving process
        """
        group.attrs['class'] = 'PoissonDistribution'
        group.attrs['scale'] = self.scale
        if save_metadata:
            self.save_metadata(group)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a `PoissonDistribution` from the given hdf5 file group.
        
        Parameters
        ----------
        group : h5py.Group
            the same hdf5 file group which fill_hdf5_group was called on when
            this Distribution was saved
        
        Returns
        -------
        distribution : `PoissonDistribution`
            distribution created from the information in the given group
        """
        try:
            assert group.attrs['class'] == 'PoissonDistribution'
        except:
            raise TypeError("The given hdf5 file doesn't seem to contain a " +\
                "PoissonDistribution.")
        metadata = Distribution.load_metadata(group)
        scale = group.attrs['scale']
        return PoissonDistribution(scale, metadata=metadata)
    
    @property
    def gradient_computable(self):
        """
        Boolean describing whether the gradient of the given distribution has
        been implemented. If True,
        `PoissonDistribution.gradient_of_log_value` method can be called
        safely.
        """
        return False 
    
    @property
    def hessian_computable(self):
        """
        Boolean describing whether the hessian of the given distribution has
        been implemented. If True,
        `PoissonDistribution.hessian_of_log_value` method can be called
        safely.
        """
        return False
    
    def copy(self):
        """
        Copies this distribution.
        
        Returns
        -------
        copied : `PoissonDistribution`
            a deep copy of this distribution, ignoring metadata.
        """
        return PoissonDistribution(self.scale)

