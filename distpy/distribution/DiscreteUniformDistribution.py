"""
Module containing class representing a discrete uniform distribution. Its PMF
is represented by: $$f(x) = \\begin{cases} \\frac{1}{b-a+1} &\
x\\in\\{a,a+1,\\ldots,b\\} \\\\ 0 & \\text{otherwise} \\end{cases}$$

**File**: $DISTPY/distpy/distribution/DiscreteUniformDistribution.py  
**Author**: Keith Tauscher  
**Date**: 31 May 2021
"""
from __future__ import division
import numpy as np
import numpy.random as rand
from ..util import int_types, sequence_types
from .Distribution import Distribution

class DiscreteUniformDistribution(Distribution):
    """
    Class representing a discrete uniform distribution. Its PMF is represented
    by: $$f(x) = \\begin{cases} \\frac{1}{b-a+1} &\
    x\\in\\{a,a+1,\\ldots,b\\} \\\\ 0 & \\text{otherwise} \\end{cases}$$
    """
    def __init__(self, low, high=0, metadata=None):
        """
        Initializes a new `DiscreteUniformDistribution` with the given
        parameter values.
        
        Parameters
        ----------
        low : int
            - if `high` is given, this is the lowest possible integer drawn by
            this `DiscreteUniformDistribution`
            - if `high` is not given, this is the highest possible integer
            drawn by this `DiscreteUniformDistribution` (with the lowest
            possible value being 0)
        high : int or None
            - if given, `high` is the highest possible integer drawn by this
            `UniformDiscreteUniformDistribution`
            - if not given, `low` is the highest possible integer drawn by this
            `DiscreteUniformDistribution` (with the lowest possible value being
            0)
        metadata : number or str or dict or `distpy.util.Savable.Savable`
            data to store alongside this distribution.
        """
        self.bounds = (low, high)
        self.metadata = metadata
    
    @property
    def bounds(self):
        """
        Tuple of the form `(min, max)` containing lowest and highest returnable
        values.
        """
        if not hasattr(self, '_bounds'):
            raise AttributeError("bounds was referenced before it was set.")
        return self._bounds
    
    @bounds.setter
    def bounds(self, value):
        """
        Setter for `DiscreteUniformDistribution.bounds`.
        
        Parameters
        ----------
        value : tuple
            tuple of form `(min, max)`, where both `min` and `max` are integers
            and `max>=min`
        """
        if type(value) in sequence_types:
            if len(value) == 2:
                if all([(type(element) in int_types) for element in value]):
                    if value[0] == value[1]:
                        raise ValueError("The lower and upper bounds were " +\
                            "set to the same number.")
                    else:
                        self._bounds = (min(value), max(value))
                else:
                    raise TypeError("Not all elements of bounds were " +\
                        "integers.")
            else:
                raise ValueError("bounds was set to a sequence of a length " +\
                    "that isn't two.")
        else:
            raise TypeError("bounds was set to a non-sequence.")
    
    @property
    def low(self):
        """
        The lowest returnable value of this `DiscreteUniformDistribution`.
        """
        return self.bounds[0]
    
    @property
    def high(self):
        """
        The highest returnable value of this `DiscreteUniformDistribution`.
        """
        return self.bounds[1]
    
    @property
    def log_probability(self):
        """
        The logarithm of the probability mass when called between
        `DiscreteUniformDistribution.low` and
        `DiscreteUniformDistribution.high`, given by \\(-\\ln{(b-a+1)}\\).
        """
        if not hasattr(self, '_log_probability'):
            self._log_probability = ((-1) * np.log(self.high - self.low + 1))
        return self._log_probability
    
    @property
    def numparams(self):
        """
        The number of parameters of this `DiscreteUniformDistribution`, 1.
        """
        return 1
    
    @property
    def mean(self):
        """
        The mean of this `DiscreteUniformDistribution`, \\(\\frac{a+b}{2}\\).
        """
        if not hasattr(self, '_mean'):
            self._mean = (self.low + self.high) / 2
        return self._mean
    
    @property
    def variance(self):
        """
        The variance of this `DiscreteUniformDistribution`,
        \\(\\frac{(b-a+1)^2-1}{12}\\).
        """
        if not hasattr(self, '_variance'):
            self._variance = (((self.high - self.low + 1) ** 2) - 1) / 12
        return self._variance
    
    def draw(self, shape=None, random=rand):
        """
        Draws point(s) from this `DiscreteUniformDistribution`.
        
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
        return random.randint(self.low, high=self.high+1, size=shape)
    
    def log_value(self, point):
        """
        Computes the logarithm of the value of this
        `DiscreteUniformDistribution` at the given point.
        
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
        if (abs(point - int(round(point))) < 1e-9) and (point >= self.low) and\
            (point <= self.high):
            return self.log_probability
        return -np.inf
    
    def to_string(self):
        """
        Finds and returns a string version of this
        `DiscreteUniformDistribution` of the form `"DiscreteUniform(a,b)"`.
        """
        return "DiscreteUniform({0:.2g}, {1:.2g})".format(self.low, self.high)
    
    def __eq__(self, other):
        """
        Checks for equality of this `DiscreteUniformDistribution` with `other`.
        
        Parameters
        ----------
        other : object
            object to check for equality
        
        Returns
        -------
        result : bool
            True if and only if `other` is a `DiscreteUniformDistribution` with
            the same `DiscreteUniformDistribution.low` and
            `DiscreteUniformDistribution.high`
        """
        if isinstance(other, DiscreteUniformDistribution):
            low_equal = (self.low == other.low)
            high_equal = (self.high == other.high)
            metadata_equal = self.metadata_equal(other)
            return all([low_equal, high_equal, metadata_equal])
        else:
            return False
    
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
        return True
    
    def fill_hdf5_group(self, group, save_metadata=True):
        """
        Fills the given hdf5 file group with data about this
        `DiscreteUniformDistribution` so that it can be loaded later.
        
        Parameters
        ----------
        group : h5py.Group
            hdf5 file group to fill
        save_metadata : bool
            - if True, attempts to save metadata alongside distribution and
            throws error if it fails
            - if False, metadata is ignored in saving process
        """
        group.attrs['class'] = 'DiscreteUniformDistribution'
        group.attrs['low'] = self.low
        group.attrs['high'] = self.high
        if save_metadata:
            self.save_metadata(group)
   
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a `DiscreteUniformDistribution` from the given hdf5 file group.
        
        Parameters
        ----------
        group : h5py.Group
            the same hdf5 file group which fill_hdf5_group was called on when
            this Distribution was saved
        
        Returns
        -------
        distribution : `DiscreteUniformDistribution`
            distribution created from the information in the given group
        """
        try:
            assert group.attrs['class'] == 'DiscreteUniformDistribution'
        except:
            raise TypeError("The given hdf5 file doesn't seem to contain a " +\
                "DiscreteUniformDistribution.")
        metadata = Distribution.load_metadata(group)
        low = group.attrs['low']
        high = group.attrs['high']
        return\
            DiscreteUniformDistribution(low=low, high=high, metadata=metadata)
    
    @property
    def gradient_computable(self):
        """
        Boolean describing whether the gradient of the given distribution has
        been implemented. If True,
        `DiscreteUniformDistribution.gradient_of_log_value` method can be
        called safely.
        """
        return False
    
    @property
    def hessian_computable(self):
        """
        Boolean describing whether the hessian of the given distribution has
        been implemented. If True,
        `DiscreteUniformDistribution.hessian_of_log_value` method can be
        called safely.
        """
        return False
    
    def copy(self):
        """
        Copies this distribution.
        
        Returns
        -------
        copied : `DiscreteUniformDistribution`
            a deep copy of this distribution, ignoring metadata.
        """
        return DiscreteUniformDistribution(self.low, self.high)

