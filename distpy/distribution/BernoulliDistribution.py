"""
Module containing class representing a Bernoulli distribution. Its PMF is
represented by: $$f(x) = \\begin{cases} p & x=1 \\\\ 1-p & x=0 \\end{cases}$$

**File**: $DISTPY/distpy/distribution/BernoulliDistribution.py  
**Author**: Keith Tauscher  
**Date**: 31 May 2021
"""
import numpy as np
import numpy.random as rand
from ..util import int_types, numerical_types
from .Distribution import Distribution

class BernoulliDistribution(Distribution):
    """
    Class representing a Bernoulli distribution. Its PMF is represented by:
    $$f(x) = \\begin{cases} p & x=1 \\\\ 1-p & x=0 \\end{cases}$$
    """
    def __init__(self, probability_of_success, metadata=None):
        """
        Initializes a new `BernoulliDistribution` with the given parameter
        values.
        
        Parameters
        ----------
        probability_of_success : float
            real number, \\(p\\), in (0, 1)
        metadata : number or str or dict or `distpy.util.Savable.Savable`
            data to store alongside this distribution.
        """
        self.probability_of_success = probability_of_success
        self.metadata = metadata
    
    @property
    def mean(self):
        """
        The mean of this `BernoulliDistribution`, \\(p\\).
        """
        if not hasattr(self, '_mean'):
            self._mean = self.probability_of_success
        return self._mean
    
    @property
    def variance(self):
        """
        The variance of this `BernoulliDistribution`, \\(p(1-p)\\).
        """
        if not hasattr(self, '_variance'):
            self._variance =\
                self.probability_of_success * (1 - self.probability_of_success)
        return self._variance
    
    @property
    def probability_of_success(self):
        """
        The probability, \\(p\\) of drawing 1 as opposed to 0.
        """
        if not hasattr(self, '_probability_of_success'):
            raise AttributeError("probability_of_success was referenced " +\
                "before it was set.")
        return self._probability_of_success
    
    @probability_of_success.setter
    def probability_of_success(self, value):
        """
        Setter for `BernoulliDistribution.probability_of_success`.
        
        Parameters
        ----------
        value : float
            real number, \\(p\\), between 0 and 1 (exclusive)
        """
        if type(value) in numerical_types:
            if (value > 0.) and (value < 1.):
                self._probability_of_success = value
            else:
                raise ValueError("probability_of_success given to " +\
                    "BinomialDistribution was not between 0 and 1.")
        else:
            raise ValueError("probability_of_success given to " +\
                "BinomialDistribution was not a number.")
    
    @property
    def numparams(self):
        """
        The number of parameters of this `BernoulliDistribution`, 1.
        """
        return 1
    
    def draw(self, shape=None, random=rand):
        """
        Draws point(s) from this `BernoulliDistribution`.
        
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
        if shape is None:
            none_shape = True
            shape = (1,)
        else:
            none_shape = False
        values = (random.uniform(size=shape) <\
            self.probability_of_success).astype(int)
        if none_shape:
            values = values[0]
        return values
    
    def log_value(self, point):
        """
        Computes the logarithm of the value of this `BernoulliDistribution` at
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
            if point == 0:
                return np.log(1 - self.probability_of_success)
            elif point == 1:
                return np.log(self.probability_of_success)
            else:
                return -np.inf
        else:
            raise TypeError("point given to BernoulliDistribution was not " +\
                "an integer.")
    
    def to_string(self):
        """
        Finds and returns a string version of this `BernoulliDistribution` of
        the form `"Bernoulli(p)"`.
        """
        return "Bernoulli({:.2g})".format(self.probability_of_success)
    
    def __eq__(self, other):
        """
        Checks for equality of this `BernoulliDistribution` with `other`.
        
        Parameters
        ----------
        other : object
            object to check for equality
        
        Returns
        -------
        result : bool
            True if and only if `other` is a `BernoulliDistribution` with the
            same `BernoulliDistribution.probability_of_success`
        """
        if isinstance(other, BernoulliDistribution):
            p_close = np.isclose(self.probability_of_success,\
                other.probability_of_success, rtol=0, atol=1e-6)
            metadata_equal = self.metadata_equal(other)
            return (p_close and metadata_equal)
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
        return 1
    
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
        `BernoulliDistribution` so that it can be loaded later.
        
        Parameters
        ----------
        group : h5py.Group
            hdf5 file group to fill
        save_metadata : bool
            - if True, attempts to save metadata alongside distribution and
            throws error if it fails
            - if False, metadata is ignored in saving process
        """
        group.attrs['class'] = 'BernoulliDistribution'
        group.attrs['probability_of_success'] = self.probability_of_success
        if save_metadata:
            self.save_metadata(group)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a `BernoulliDistribution` from the given hdf5 file group.
        
        Parameters
        ----------
        group : h5py.Group
            the same hdf5 file group which fill_hdf5_group was called on when
            this Distribution was saved
        
        Returns
        -------
        distribution : `BernoulliDistribution`
            distribution created from the information in the given group
        """
        try:
            assert group.attrs['class'] == 'BernoulliDistribution'
        except:
            raise TypeError("The given hdf5 file doesn't seem to contain a " +\
                "BernoulliDistribution.")
        metadata = Distribution.load_metadata(group)
        probability_of_success = group.attrs['probability_of_success']
        return\
            BernoulliDistribution(probability_of_success, metadata=metadata)
    
    @property
    def gradient_computable(self):
        """
        Boolean describing whether the gradient of the given distribution has
        been implemented. If True,
        `BernoulliDistribution.gradient_of_log_value` method can be called
        safely.
        """
        return False 
    
    @property
    def hessian_computable(self):
        """
        Boolean describing whether the hessian of the given distribution has
        been implemented. If True,
        `BernoulliDistribution.hessian_of_log_value` method can be called
        safely.
        """
        return False
    
    def copy(self):
        """
        Copies this distribution.
        
        Returns
        -------
        copied : `BernoulliDistribution`
            a deep copy of this distribution, ignoring metadata.
        """
        return BernoulliDistribution(self.probability_of_success)

