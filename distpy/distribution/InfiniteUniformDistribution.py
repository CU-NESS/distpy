"""
Module containing class representing an improper distribution that is equally
likely to be anywhere. It is improper because it cannot be normalized.

**File**: $DISTPY/distpy/distribution/InfiniteUniformDistribution.py  
**Author**: Keith Tauscher  
**Date**: 31 May 2021
"""
import numpy as np
from ..util import int_types, bool_types, numerical_types, sequence_types
from .Distribution import Distribution

class InfiniteUniformDistribution(Distribution):
    """
    Class representing an improper distribution that is equally likely to be
    anywhere. It is improper because it cannot be normalized.
    """
    def __init__(self, ndim=1, minima=None, maxima=None,\
        is_discrete=False, metadata=None):
        """
        Initializes a new `InfiniteUniformDistribution` with the given
        parameter values.
        
        Parameters
        ----------
        ndim : int
            positive integer number of dimensions this improper distribution
            applies to
        minima : None or sequence
            the minimum possible value of each parameter (`minima` and `maxima`
            cannot be such that a variable has both, because then the
            distribution would not be improper and could be handled correctly)
        maxima : None or sequence
            the maximum possible value of each parameter (`minima` and `maxima`
            cannot be such that a variable has both, because then the
            distribution would not be improper and could be handled correctly)
        is_discrete : bool
            bool determining whether or not this distribution is considered
            discrete
        metadata : number or str or dict or `distpy.util.Savable.Savable`
            data to store alongside this distribution.
        """
        self.numparams = ndim
        self.minima = minima
        self.maxima = maxima
        self.is_discrete = is_discrete
        self.metadata = metadata
    
    @property
    def minima(self):
        """
        The minimum variable value(s) in an array.
        """
        if not hasattr(self, '_minima'):
            raise AttributeError("minima was referenced before it was set.")
        return self._minima
    
    @minima.setter
    def minima(self, value):
        """
        Setter for `InfiniteUniformDistribution.minima`.
        
        value : float or sequence or None
            - if None, no lower bound is used for any parameters
            - if a number, that is the number used as a lower bound for all
            parameters
            - otherwise (only if numparams > 1), must be a sequence of length
            `InfiniteUniformDistribution.numparams` containing None's and/or
            numbers containing lower bounds
        """
        if self.numparams == 1:
            if type(value) is type(None):
                self._minima = -np.inf
            elif type(value) in numerical_types:
                self._minima = value
            else:
                raise TypeError("minima was set to neither None nor a number.")
        elif type(value) is type(None):
            self._minima = np.ones((self.numparams,)) * (-np.inf)
        elif type(value) in numerical_types:
            self._minima = np.ones((self.numparams,)) * value
        elif type(value) in sequence_types:
            if len(value) == self.numparams:
                self._minima = np.array([((-np.inf)\
                    if (type(element) is type(None)) else element)\
                    for element in value])
            else:
                raise ValueError("The sequence of minima given to an " +\
                    "InfiniteUniformDistribution object was not of length " +\
                    "ndim.")
        else:
            raise TypeError("minima was set to neither None nor a number " +\
                "or sequence.")
    
    @property
    def maxima(self):
        """
        The maximum variable value(s) in an array.
        """
        if not hasattr(self, '_maxima'):
            raise AttributeError("maxima was referenced before it was set.")
        return self._maxima
    
    @maxima.setter
    def maxima(self, value):
        """
        Setter for `InfiniteUniformDistribution.maxima`.
        
        value : float or sequence or None
            - if None, no lower bound is used for any parameters
            - if a number, that is the number used as a upper bound for all
            parameters
            - otherwise (only if numparams > 1), must be a sequence of length
            `InfiniteUniformDistribution.numparams` containing None's and/or
            numbers containing upper bounds
        """
        if self.numparams == 1:
            if type(value) is type(None):
                self._maxima = np.inf
            elif type(value) in numerical_types:
                self._maxima = value
            else:
                raise TypeError("maxima was set to neither None nor a number.")
        elif type(value) is type(None):
            self._maxima = np.ones((self.numparams,)) * (np.inf)
        elif type(value) in numerical_types:
            self._maxima = np.ones((self.numparams,)) * value
        elif type(value) in sequence_types:
            if len(value) == self.numparams:
                self._maxima = np.array([(np.inf\
                    if (type(element) is type(None)) else element)\
                    for element in value])
            else:
                raise ValueError("The sequence of maxima given to an " +\
                    "InfiniteUniformDistribution object was not of length " +\
                    "ndim.")
        else:
            raise TypeError("maxima was set to neither None nor a number " +\
                "or sequence.")
        if np.any(np.all(np.isfinite(np.array([self.minima, self._maxima])),\
            axis=0)):
            raise ValueError("At least one parameter of an " +\
                "InfiniteUniformDistribution had both upper and lower " +\
                "bounds, meaning it would be better for it to be " +\
                "encapsulated with a regular UniformDistribution instead " +\
                "of an InfiniteUniformDistribution.")
    
    @property
    def all_left_infinite(self):
        """
        Boolean describing whether all parameters of this distribution have no
        lower bound.
        """
        if not hasattr(self, '_all_left_infinite'):
            self._all_left_infinite = (not np.any(np.isfinite(self.minima)))
        return self._all_left_infinite
    
    @property
    def all_right_infinite(self):
        """
        Boolean describing whether all parameters of this distribution have no
        upper bound.
        """
        if not hasattr(self, '_all_right_infinite'):
            self._all_right_infinite = (not np.any(np.isfinite(self.maxima)))
        return self._all_right_infinite
    
    @property
    def all_doubly_infinite(self):
        """
        Boolean describing whether this distribution is doubly infinite in all
        of its parameters. In this case, no comparison is done when this
        distribution is called.
        """
        if not hasattr(self, '_all_doubly_infinite'):
            self._all_doubly_infinite =\
                (self.all_left_infinite and self.all_right_infinite)
        return self._all_doubly_infinite
    
    def draw(self, shape=None, random=None):
        """
        Draws a point from the distribution. Since
        `InfiniteUniformDistribution` cannot be drawn from, this throws a
        NotImplementedError.
        """
        raise NotImplementedError("InfiniteUniformDistribution objects " +\
            "cannot be drawn from because there is zero probability of its " +\
            "variate appearing in any given finite interval.")
    
    def log_value(self, point):
        """
        Computes the logarithm of the value of this
        `InfiniteUniformDistribution` at the given point.
        
        Parameters
        ----------
        point : float or `numpy.ndarray`
            - if this distribution is univariate, `point` should be a scalar
            - if this distribution describes \\(p\\) parameters, `point` should
            be a length-\\(p\\) `numpy.ndarray`
        
        Returns
        -------
        value : float
            0 if `point` is valid, -np.inf otherwise
        """
        if ((self.all_left_infinite or np.all(point > self.minima)) and\
            (self.all_right_infinite or np.all(point < self.maxima))):
            return 0.
        else:
            return (-np.inf)
    
    @property
    def gradient_computable(self):
        """
        Boolean describing whether the gradient of the given distribution has
        been implemented. If True,
        `InfiniteUniformDistribution.gradient_of_log_value` method can be
        called safely.
        """
        return True
    
    def gradient_of_log_value(self, point):
        """
        Computes the gradient (derivative) of the logarithm of the value of
        this `InfiniteUniformDistribution` at the given point.
        
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
        if self.numparams == 1:
            return 0.
        else:
            return np.zeros(self.numparams)
    
    @property
    def hessian_computable(self):
        """
        Boolean describing whether the hessian of the given distribution has
        been implemented. If True,
        `InfiniteUniformDistribution.hessian_of_log_value` method can be
        called safely.
        """
        return True
    
    def hessian_of_log_value(self, point):
        """
        Computes the hessian (second derivative) of the logarithm of the value
        of this `InfiniteUniformDistribution` at the given point.
        
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
        if self.numparams:
            return 0.
        else:
            return np.zeros((self.numparams, self.numparams))
    
    @property
    def numparams(self):
        """
        The number of parameters of this `InfiniteUniformDistribution`.
        """
        if not hasattr(self, '_numparams'):
            raise AttributeError("numparams referenced before it was set.")
        return self._numparams
    
    @property
    def mean(self):
        """
        There is no mean of `InfiniteUniformDistribution`.
        """
        if not hasattr(self, '_mean'):
            raise AttributeError("mean is not defined for the " +\
                "InfiniteUniformDistribution.")
        return self._mean
    
    @property
    def variance(self):
        """
        There is no variance of `InfiniteUniformDistribution`.
        """
        if not hasattr(self, '_variance'):
            raise NotImplementedError("variance is not defined for the " +\
                "InfiniteUniformDistribution.")
        return self._variance
    
    @numparams.setter
    def numparams(self, value):
        """
        Setter for `InfiniteUniformDistribution.numparams`.
        
        Parameters
        ----------
        value : int
            a positive integer
        """
        if type(value) in int_types:
            if value > 0:
                self._numparams = value
            else:
                raise ValueError("numparams was set to a non-positive " +\
                    "integer.")
        else:
            raise TypeError("numparams was set to a non-integer.")
    
    def to_string(self):
        """
        Finds and returns a string version of this
        `InfiniteUniformDistribution` of the form `"InfiniteUniform"`.
        """
        return 'InfiniteUniform'
    
    def __eq__(self, other):
        """
        Checks for equality of this `InfiniteUniformDistribution` with `other`.
        
        Parameters
        ----------
        other : object
            object to check for equality
        
        Returns
        -------
        result : bool
            True if and only if `other` is a `InfiniteUniformDistribution` with
            the same `InfiniteUniformDistribution.is_discrete`,
            `InfiniteUniformDistribution.minima`, and
            `InfiniteUniformDistribution.maxima`
        """
        if not isinstance(other, InfiniteUniformDistribution):
            return False
        if self.is_discrete != other.is_discrete:
            return False
        if np.any(self.minima != other.minima):
            return False
        if np.any(self.maxima != other.maxima):
            return False
        return self.metadata_equal(other)
    
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
        Setter for `InfiniteUniformDistribution.is_discrete`
        
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
        `InfiniteUniformDistribution` so that it can be loaded later.
        
        Parameters
        ----------
        group : h5py.Group
            hdf5 file group to fill
        save_metadata : bool
            - if True, attempts to save metadata alongside distribution and
            throws error if it fails
            - if False, metadata is ignored in saving process
        """
        group.attrs['class'] = 'InfiniteUniformDistribution'
        group.attrs['is_discrete'] = self.is_discrete
        group.attrs['ndim'] = self.numparams
        group.attrs['minima'] = self.minima
        group.attrs['maxima'] = self.maxima
        if save_metadata:
            self.save_metadata(group)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a `InfiniteUniformDistribution` from the given hdf5 file group.
        
        Parameters
        ----------
        group : h5py.Group
            the same hdf5 file group which fill_hdf5_group was called on when
            this Distribution was saved
        
        Returns
        -------
        distribution : `InfiniteUniformDistribution`
            distribution created from the information in the given group
        """
        try:
            assert group.attrs['class'] == 'InfiniteUniformDistribution'
        except:
            raise TypeError("The given hdf5 file doesn't seem to contain a " +\
                "InfiniteUniformDistribution.")
        metadata = Distribution.load_metadata(group)
        ndim = group.attrs['ndim']
        if 'is_discrete' in group.attrs:
            is_discrete = group.attrs['is_discrete']
        else:
            is_discrete = False
        if 'minima' in group.attrs:
            minima = group.attrs['minima']
        else:
            minima = None
        if 'maxima' in group.attrs:
            maxima = group.attrs['maxima']
        else:
            maxima = None
        return InfiniteUniformDistribution(ndim, minima=minima, maxima=maxima,\
            is_discrete=is_discrete, metadata=metadata)
    
    @property
    def minimum(self):
        """
        The minimum allowable value(s) in this distribution.
        """
        if self.numparams == 1:
            return (self.minima if np.isfinite(self.minima) else None)
        else:
            return [(element if np.isfinite(element) else None)\
                for element in self.minima]
    
    @property
    def maximum(self):
        """
        The maximum allowable value(s) in this distribution.
        """
        if self.numparams == 1:
            return (self.maxima if np.isfinite(self.maxima) else None)
        else:
            return [(element if np.isfinite(element) else None)\
                for element in self.maxima]
    
    @property
    def can_give_confidence_intervals(self):
        """
        Improper distributions do not support confidence intervals.
        """
        return False
    
    def copy(self):
        """
        Copies this distribution.
        
        Returns
        -------
        copied : `InfiniteUniformDistribution`
            a deep copy of this distribution, ignoring metadata.
        """
        return InfiniteUniformDistribution(self.numparams,\
            minima=self.minima, maxima=self.maxima,\
            is_discrete=self.is_discrete)

