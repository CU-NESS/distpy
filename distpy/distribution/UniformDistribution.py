"""
Module containing class representing a uniform distribution. Its PDF is
represented by: $$f(x) = \\begin{cases} \\frac{1}{b-a} & a\\le x\\le b \\\\\
0 & \\text{otherwise} \\end{cases}$$

**File**: $DISTPY/distpy/distribution/UniformDistribution.py  
**Author**: Keith Tauscher  
**Date**: 31 May 2021
"""
from __future__ import division
import numpy as np
import numpy.random as rand
from ..util import numerical_types, sequence_types
from .Distribution import Distribution

class UniformDistribution(Distribution):
    """
    Class representing a uniform distribution. Its PDF is represented by:
    $$f(x) = \\begin{cases} \\frac{1}{b-a} & a\\le x\\le b \\\\ 0 &\
    \\text{otherwise} \\end{cases}$$
    """
    def __init__(self, low=0., high=1., metadata=None):
        """
        Initializes a new `UniformDistribution` with the given parameter
        values.
        
        Parameters
        ----------
        low : float
            real number, \\(a\\)
        high : float
            real number, \\(b\\), that is greater than \\(a\\)
        metadata : number or str or dict or `distpy.util.Savable.Savable`
            data to store alongside this distribution.
        """
        self.bounds = (low, high)
        self.metadata = metadata
    
    @property
    def bounds(self):
        """
        The lower and upper bounds of this distribution in a tuple.
        """
        if not hasattr(self, '_bounds'):
            raise AttributeError("bounds was referenced before it was set.")
        return self._bounds
    
    @bounds.setter
    def bounds(self, value):
        """
        Setter for `UniformDistribution.bounds`.
        
        Parameters
        ----------
        value : tuple
            2-tuple of form `(lower_bound, upper_bound)`
        """
        if type(value) in sequence_types:
            if len(value) == 2:
                if all([(type(element) in numerical_types)\
                    for element in value]):
                    if value[0] == value[1]:
                        raise ValueError("The lower bound and upper bound " +\
                            "were set to the same number.")
                    else:
                        self._bounds = (min(value), max(value))
                else:
                    raise TypeError("At least one element of bounds was " +\
                        "not a number.")
            else:
                raise ValueError("bounds was set to a sequence of a length " +\
                    "other than two.")
        else:
            raise TypeError("bounds was set to a non-sequence.")
    
    @property
    def low(self):
        """
        The lower bound of this distribution.
        """
        return self.bounds[0]
    
    @property
    def high(self):
        """
        The upper bound of this distribution.
        """
        return self.bounds[1]
    
    @property
    def log_probability(self):
        """
        The log of the probability density inside the domain, given by
        \\(-\\ln{(b-a)}\\).
        """
        if not hasattr(self, '_log_probability'):
            self._log_probability = ((-1) * np.log(self.high - self.low))
        return self._log_probability

    @property
    def numparams(self):
        """
        The number of parameters of this `UniformDistribution`, 1.
        """
        return 1
    
    @property
    def mean(self):
        """
        The mean of this `UniformDistribution`, \\(\\frac{a+b}{2}\\).
        """
        if not hasattr(self, '_mean'):
            self._mean = (self.low + self.high) / 2
        return self._mean
    
    @property
    def variance(self):
        """
        The variance of this `UniformDistribution`, \\(\\frac{(b-a)^2}{12}\\).
        """
        if not hasattr(self, '_variance'):
            self._variance = ((self.high - self.low) ** 2) / 12
        return self._variance

    def draw(self, shape=None, random=rand):
        """
        Draws point(s) from this `UniformDistribution`.
        
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
        return random.uniform(low=self.low, high=self.high, size=shape)


    def log_value(self, point):
        """
        Computes the logarithm of the value of this `UniformDistribution` at
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
        if (point >= self.low) and (point <= self.high):
            return self.log_probability
        return -np.inf
    
    def to_string(self):
        """
        Finds and returns a string version of this `UniformDistribution` of the
        form `"Uniform(a,b)"`.
        """
        return "Uniform({0:.2g}, {1:.2g})".format(self.low, self.high)
    
    def __eq__(self, other):
        """
        Checks for equality of this `UniformDistribution` with `other`.
        
        Parameters
        ----------
        other : object
            object to check for equality
        
        Returns
        -------
        result : bool
            True if and only if `other` is a `UniformDistribution` with the
            same `UniformDistribution.low`, `UniformDistribution.high`
        """
        if isinstance(other, UniformDistribution):
            tol_kwargs = {'rtol': 0., 'atol': 1e-9}
            low_close = np.isclose(self.low, other.low, **tol_kwargs)
            high_close = np.isclose(self.high, other.high, **tol_kwargs)
            metadata_equal = self.metadata_equal(other)
            return all([low_close, high_close, metadata_equal])
        else:
            return False
    
    def inverse_cdf(self, cdf):
        """
        Computes the inverse of the cumulative distribution function (cdf) of
        this `UniformDistribution`.
        
        Parameters
        ----------
        cdf : float
            probability value between 0 and 1
        
        Returns
        -------
        point : float
            value which yields `cdf` when it the CDF is evaluated at it
        """
        return (self.low + ((self.high - self.low) * cdf))
    
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
        `UniformDistribution` so that it can be loaded later.
        
        Parameters
        ----------
        group : h5py.Group
            hdf5 file group to fill
        save_metadata : bool
            - if True, attempts to save metadata alongside distribution and
            throws error if it fails
            - if False, metadata is ignored in saving process
        """
        group.attrs['class'] = 'UniformDistribution'
        group.attrs['low'] = self.low
        group.attrs['high'] = self.high
        if save_metadata:
            self.save_metadata(group)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a `UniformDistribution` from the given hdf5 file group.
        
        Parameters
        ----------
        group : h5py.Group
            the same hdf5 file group which fill_hdf5_group was called on when
            this Distribution was saved
        
        Returns
        -------
        distribution : `UniformDistribution`
            distribution created from the information in the given group
        """
        try:
            assert group.attrs['class'] == 'UniformDistribution'
        except:
            raise TypeError("The given hdf5 file doesn't seem to contain a " +\
                "UniformDistribution.")
        metadata = Distribution.load_metadata(group)
        low = group.attrs['low']
        high = group.attrs['high']
        return UniformDistribution(low=low, high=high, metadata=metadata)
    
    @property
    def gradient_computable(self):
        """
        Boolean describing whether the gradient of the given distribution has
        been implemented. If True,
        `UniformDistribution.gradient_of_log_value` method can be called
        safely.
        """
        return True
    
    def gradient_of_log_value(self, point):
        """
        Computes the gradient (derivative) of the logarithm of the value of
        this `UniformDistribution` at the given point.
        
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
        return 0.
    
    @property
    def hessian_computable(self):
        """
        Boolean describing whether the hessian of the given distribution has
        been implemented. If True,
        `UniformDistribution.hessian_of_log_value` method can be called
        safely.
        """
        return True
    
    def hessian_of_log_value(self, point):
        """
        Computes the hessian (second derivative) of the logarithm of the value
        of this `UniformDistribution` at the given point.
        
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
        copied : `UniformDistribution`
            a deep copy of this distribution, ignoring metadata.
        """
        return UniformDistribution(self.low, self.high)

