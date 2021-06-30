"""
Module containing class representing a delta distribution. Its PDF is
represented by: $$f(x) = \\delta(x-\\mu),$$ where \\(\\delta(x)\\) is the Dirac
delta function.

**File**: $DISTPY/distpy/distribution/KroneckerDeltaDistribution.py  
**Author**: Keith Tauscher  
**Date**: 31 May 2021
"""
import numpy as np
from ..util import int_types, numerical_types, sequence_types, bool_types
from .Distribution import Distribution

class KroneckerDeltaDistribution(Distribution):
    """
    Class representing a delta distribution. Its PDF is represented by:
    $$f(x) = \\delta(x-\\mu),$$ where \\(\\delta(x)\\) is the Dirac delta
    function.
    """
    def __init__(self, value, is_discrete=True, metadata=None):
        """
        Initializes a new `KroneckerDeltaDistribution` with the given parameter
        values.
        
        Parameters
        ----------
        value : float or `numpy.ndarray`
            the value, \\(\\mu\\), that is always returned
        is_discrete : bool
            bool determining whether this distribution is considered discrete
        metadata : number or str or dict or `distpy.util.Savable.Savable`
            data to store alongside this distribution.
        """
        self.value = value
        self.is_discrete = is_discrete
        self.metadata = metadata
    
    @property
    def value(self):
        """
        The value which is always returned by this distribution.
        """
        if not hasattr(self, '_value'):
            raise AttributeError("value referenced before it was set.")
        return self._value
    
    @value.setter
    def value(self, value):
        """
        Setter for `KroneckerDeltaDistribution.value`
        
        Parameters
        ----------
        value : int or float or numpy.ndarray
            value which is always returned by this distribution
        """
        if type(value) in numerical_types:
            self._value = value
        elif type(value) in sequence_types:
            value = np.array(value)
            if value.ndim == 1:
                if value.size == 1:
                    self._value = value[0]
                else:
                    self._value = value
            else:
                raise ValueError("KroneckerDeltaDistribution must be " +\
                    "initialized with either a number value or a non-empty " +\
                    "1D numpy.ndarray value.")
        else:
            raise TypeError("value was set to a non-number.")
    
    @property
    def mean(self):
        """
        The mean of this `KroneckerDeltaDistribution`, \\(\\mu\\).
        """
        if not hasattr(self, '_mean'):
            self._mean = self.value
        return self._mean
    
    @property
    def variance(self):
        """
        The variance of this `KroneckerDeltaDistribution`, \\(0\\).
        """
        if not hasattr(self, '_variance'):
            if self.numparams == 1:
                self._variance = 0
            else:
                self._variance = np.zeros(2 * (self.numparams,))
        return self._variance
    
    def draw(self, shape=None, random=None):
        """
        Draws point(s) from this `KroneckerDeltaDistribution`.
        
        Parameters
        ----------
        shape : int or tuple or None
            - if None, returns single random variate:
                - if this distribution is univariate, a scalar is returned
                - if this distribution describes \\(p\\) parameters, then a 1D
                array of length \\(p\\) is returned
            - if int, \\(n\\), returns \\(n\\) random variates:
                - if this distribution is univariate, a 1D array of length
                \\(n\\) is returned
                - if this distribution describes \\(p\\) parameters, then a 2D
                array of shape `(n,p)` is returned
            - if tuple of \\(n\\) ints, returns `numpy.prod(shape)` random
            variates:
                - if this distribution is univariate, an \\(n\\)-D array of
                shape `shape` is returned
                - if this distribution describes \\(p\\) parameters, then an
                \\((n+1)\\)-D array of shape `shape+(p,)` is returned
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
            return self.value
        else:
            if type(shape) in int_types:
                shape = (shape,)
            return_value = self.value * np.ones(shape + (self.numparams,))
            if self.numparams == 1:
                return return_value[...,0]
            else:
                return return_value
    
    def log_value(self, point):
        """
        Computes the logarithm of the value of this
        `KroneckerDeltaDistribution` at the given point.
        
        Parameters
        ----------
        point : float or `numpy.ndarray`
            - if this distribution is univariate, `point` should be a scalar
            - if this distribution describes \\(p\\) parameters, `point` should
            be a length-\\(p\\) `numpy.ndarray`
        
        Returns
        -------
        value : float
            0 if `point` is `KroneckerDeltaDistribution.value`, -np.inf
            otherwise
        """
        if np.all(point == self.value):
            return 0.
        else:
            return -np.inf
    
    @property
    def gradient_computable(self):
        """
        Boolean describing whether the gradient of the given distribution has
        been implemented. If True,
        `KroneckerDeltaDistribution.gradient_of_log_value` method can be called
        safely.
        """
        return True
    
    def gradient_of_log_value(self, point):
        """
        Computes the gradient (derivative) of the logarithm of the value of
        this `KroneckerDeltaDistribution` at the given point.
        
        Parameters
        ----------
        point : float or `numpy.ndarray`
            - if this distribution is univariate, `point` should be a scalar
            - if this distribution describes \\(p\\) parameters, `point` should
            be a length-\\(p\\) `numpy.ndarray`
        
        Returns
        -------
        value : float or `numpy.ndarray`
            gradient of the natural logarithm of the value of this
            distribution. If \\(f\\) is this distribution's PDF and \\(x\\) is
            `point`, then `value` is
            \\(\\boldsymbol{\\nabla}\\ln{\\big(f(x)\\big)}\\):
            
            - if this distribution is univariate, then a float representing the
            derivative is returned
            - if this distribution describes \\(p\\) parameters, then a 1D
            `numpy.ndarray` of length \\(p\\) is returned
        """
        return (point * 0)
    
    @property
    def hessian_computable(self):
        """
        Boolean describing whether the hessian of the given distribution has
        been implemented. If True,
        `KroneckerDeltaDistribution.hessian_of_log_value` method can be called
        safely.
        """
        return True
    
    def hessian_of_log_value(self, point):
        """
        Computes the hessian (second derivative) of the logarithm of the value
        of this `KroneckerDeltaDistribution` at the given point.
        
        Parameters
        ----------
        point : float or `numpy.ndarray`
            - if this distribution is univariate, `point` should be a scalar
            - if this distribution describes \\(p\\) parameters, `point` should
            be a length-\\(p\\) `numpy.ndarray`
        
        Returns
        -------
        value : float or `numpy.ndarray`
            hessian of the natural logarithm of the value of this
            distribution. If \\(f\\) is this distribution's PDF and \\(x\\) is
            `point`, then `value` is \\(\\boldsymbol{\\nabla}\
            \\boldsymbol{\\nabla}^T\\ln{\\big(f(x)\\big)}\\):
            
            - if this distribution is univariate, then a float representing the
            derivative is returned
            - if this distribution describes \\(p\\) parameters, then a 2D
            `numpy.ndarray` that is \\(p\\times p\\) is returned
        """
        return (point * 0)
    
    @property
    def numparams(self):
        """
        The number of parameters of this `KroneckerDeltaDistribution`.
        """
        if not hasattr(self, '_numparams'):
            if type(self.value) in numerical_types:
                self._numparams = 1
            else:
                self._numparams = len(self.value)
        return self._numparams
    
    def to_string(self):
        """
        Finds and returns a string version of this `KroneckerDeltaDistribution`
        of the form `"KroneckerDeltaDistribution(mu)"`.
        """
        return 'KroneckerDeltaDistribution({!s})'.format(self.value)
    
    def __eq__(self, other):
        """
        Checks for equality of this `KroneckerDeltaDistribution` with `other`.
        
        Parameters
        ----------
        other : object
            object to check for equality
        
        Returns
        -------
        result : bool
            True if and only if `other` is a `KroneckerDeltaDistribution` with
            the same `KroneckerDeltaDistribution.value`
        """
        if isinstance(other, KroneckerDeltaDistribution):
            value_equal = np.all(self.value == other.value)
            metadata_equal = self.metadata_equal(other)
            return (value_equal and metadata_equal)
        else:
            return False
    
    @property
    def minimum(self):
        """
        The minimum allowable value(s) in this distribution.
        """
        return self.value
    
    @property
    def maximum(self):
        """
        The maximum allowable value(s) in this distribution.
        """
        return self.value
    
    @property
    def is_discrete(self):
        """
        Boolean describing whether this distribution is discrete (True) or
        continuous (False).
        """
        if not hasattr(self, '_is_discrete'):
            raise AttributeError("is_discrete referenced before it was set.")
        return self._is_discrete
    
    @is_discrete.setter
    def is_discrete(self, value):
        """
        Setter for `KroneckerDeltaDistribution.is_discrete`.
        
        Parameters
        ----------
        value : bool
            True or False
        """
        if type(value) in bool_types:
            self._is_discrete = value
        else:
            raise TypeError("is_discrete was set to a non-bool.")
    
    def fill_hdf5_group(self, group, save_metadata=True):
        """
        Fills the given hdf5 file group with data about this
        `KroneckerDeltaDistribution` so that it can be loaded later.
        
        Parameters
        ----------
        group : h5py.Group
            hdf5 file group to fill
        save_metadata : bool
            - if True, attempts to save metadata alongside distribution and
            throws error if it fails
            - if False, metadata is ignored in saving process
        """
        group.attrs['class'] = 'KroneckerDeltaDistribution'
        group.attrs['value'] = self.value
        if save_metadata:
            self.save_metadata(group)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a `KroneckerDeltaDistribution` from the given hdf5 file group.
        
        Parameters
        ----------
        group : h5py.Group
            the same hdf5 file group which fill_hdf5_group was called on when
            this Distribution was saved
        
        Returns
        -------
        distribution : `KroneckerDeltaDistribution`
            distribution created from the information in the given group
        """
        try:
            assert group.attrs['class'] == 'KroneckerDeltaDistribution'
        except:
            raise TypeError("The given hdf5 file doesn't seem to contain a " +\
                "KroneckerDeltaDistribution.")
        metadata = Distribution.load_metadata(group)
        value = group.attrs['value']
        return KroneckerDeltaDistribution(value, metadata=metadata)
    
    @property
    def confidence_interval(self):
        """
        Confidence interval as a 2-tuple of the form (value, value). It is
        returned by all confidence interval functions.
        """
        return (self.value, self.value)
    
    def left_confidence_interval(self, probability_level):
        """
        Finds confidence interval furthest to the left.
        
        Parameters
        ----------
        probability_level : float
            the probability with which a random variable with this distribution
            will exist in returned interval
        
        Returns
        -------
        interval : tuple
            interval in a 2-tuple of form (low, high)
        """
        return self.confidence_interval
    
    def central_confidence_interval(self, probability_level):
        """
        Finds confidence interval with equal amounts of probability on each
        side.
        
        Parameters
        ----------
        probability_level : float
            the probability with which a random variable with this distribution
            will exist in returned interval
        
        Returns
        -------
        interval : tuple
            interval in a 2-tuple of form (low, high)
        """
        return self.confidence_interval
    
    def right_confidence_interval(self, probability_level):
        """
        Finds confidence interval furthest to the right.
        
        Parameters
        ----------
        probability_level : float
            the probability with which a random variable with this distribution
            will exist in returned interval
        
        Returns
        -------
        interval : tuple
            interval in a 2-tuple of form (low, high)
        """
        return self.confidence_interval
    
    def copy(self):
        """
        Copies this distribution.
        
        Returns
        -------
        copied : `KroneckerDeltaDistribution`
            a deep copy of this distribution, ignoring metadata.
        """
        return KroneckerDeltaDistribution(self.value,\
            is_discrete=self.is_discrete)

