"""
Module containing class representing a distribution where all parameters are
equal and can follow any univariate distribution. Its PDF is represented by:
$$f(x_1,x_2,\\ldots,x_N)=g(x_1)\\prod_{k=2}^N\\delta(x_k-x_1),$$ where \\(g\\)
is the PDF (or PMF) of a univariate distribution.

**File**: $DISTPY/distpy/distribution/LinkedDistribution.py  
**Author**: Keith Tauscher  
**Date**: 31 May 2021
"""
import numpy as np
import numpy.random as rand
from ..util import int_types, numerical_types, sequence_types
from .Distribution import Distribution

class LinkedDistribution(Distribution):
    """
    Class representing a distribution where all parameters are equal and can
    follow any univariate distribution. Its PDF is represented by:
    $$f(x_1,x_2,\\ldots,x_N)=g(x_1)\\prod_{k=2}^N\\delta(x_k-x_1),$$ where
    \\(g\\) is the PDF (or PMF) of a univariate distribution.
    """
    def __init__(self, shared_distribution, numparams, metadata=None):
        """
        Initializes a new `LinkedDistribution` with the given parameter values.
        
        Parameters
        ----------
        shared_distribution : `distpy.distribution.Distribution.Distribution`
            the distribution, with PDF \\(g\\), of the shared value
        numparams : int
            the integer, \\(n>2\\), number of parameters of this distribution
        metadata : number or str or dict or `distpy.util.Savable.Savable`
            data to store alongside this distribution.
        """
        self.shared_distribution = shared_distribution
        self.numparams = numparams
        self.metadata = metadata
    
    @property
    def shared_distribution(self):
        """
        The distribution shared by all of the parameters.
        """
        if not hasattr(self, '_shared_distribution'):
            raise AttributeError("shared_distribution was referenced " +\
                "before it was set.")
        return self._shared_distribution
    
    @shared_distribution.setter
    def shared_distribution(self, value):
        """
        Setter for `LinkedDistirbution.shared_distribution`.
        
        Parameters
        ----------
        value : `distpy.distribution.Distribution.Distribution`
            a univariate distribution
        """
        if isinstance(value, Distribution):
            if value.numparams == 1:
                self._shared_distribution = value
            else:
                raise NotImplementedError("The shared_distribution " +\
                    "provided to a LinkedDistribution was multivariate (I " +\
                    "don't know how to deal with this).")
        else:
            raise ValueError("The shared_distribution given to a " +\
                "LinkedDistribution was not recognizable as a distribution.")

    @property
    def numparams(self):
        """
        The number of parameters of this `LinkedDistribution`.
        """
        if not hasattr(self, '_numparams'):
            raise AttributeError("numparams was referenced before it was set.")
        return self._numparams
    
    @property
    def mean(self):
        """
        The mean of this `LinkedDistribution`, which is a
        `LinkedDistribution.numparams`-length array containing copies of the
        mean of the `LinkedDistribution.shared_distribution`.
        """
        if not hasattr(self, '_mean'):
            self._mean =\
                self.shared_distribution.mean * np.ones((self.numparams,))
        return self._mean
    
    @property
    def variance(self):
        """
        The (singular) covariance of this `LinkedDistribution`.
        """
        if not hasattr(self, '_variance'):
            self._variance = self.shared_distribution.variance *\
                np.ones(2 * (self.numparams,))
        return self._variance
    
    @numparams.setter
    def numparams(self, value):
        """
        Setter for `LinkedDistribution.numparams`.
        
        Parameters
        ----------
        value : int
            positive integer
        """
        if (type(value) in numerical_types):
            if value > 1:
                self._numparams = value
            else:
                raise ValueError("A LinkedDistribution was initialized " +\
                    "with only one parameter. Is this really what you want?")
        else:
            raise ValueError("The type of the number of parameters given " +\
                "to a LinkedDistribution was not numerical.")

    def draw(self, shape=None, random=rand):
        """
        Draws point(s) from this `LinkedDistribution`. Below, `p` is
        `LinkedDistribution.numparams`.
        
        Parameters
        ----------
        shape : int or tuple or None
            - if None, returns single random variate as a 1D array of length
            `p` is returned
            - if int, \\(n\\), returns \\(n\\) random variates as a 2D
            array of shape `(n,p)` is returned
            - if tuple of \\(n\\) ints, returns `numpy.prod(shape)` random
            variates as an \\((n+1)\\)-D array of shape `shape+(p,)` is
            returned
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
            return np.ones(self.numparams) *\
                self.shared_distribution.draw(random=random)
        else:
            if type(shape) in int_types:
                shape = (shape,)
            return np.ones(shape + (self.numparams,)) *\
                self.shared_distribution.draw(shape=shape,\
                random=random)[...,np.newaxis]

    def log_value(self, point):
        """
        Computes the logarithm of the value of this `LinkedDistribution` at the
        given point.
        
        Parameters
        ----------
        point : `numpy.ndarray`
            if this distribution describes \\(p\\) parameters, `point` should
            be a length-\\(p\\) `numpy.ndarray`
        
        Returns
        -------
        value : float
            natural logarithm of the value of this distribution at `point`. If
            \\(f\\) is this distribution's PDF and \\(x\\) is `point`, then
            `value` is \\(\\ln{\\big(f(x)\\big)}\\)
        """
        if type(point) in numerical_types:
            return self.shared_distribution.log_value(point)
        elif type(point) in sequence_types:
            if (len(point) == self.numparams):
                for ival in range(len(point)):
                    if point[ival] != point[0]:
                        return -np.inf
                return self.shared_distribution.log_value(point[0])
            else:
                raise ValueError("The length of the point given to a " +\
                    "LinkedDistribution was not the same as the " +\
                    "LinkedDistribution's number of parameters.")
        else:
            raise ValueError("The point provided to a LinkedDistribution " +\
                "was not of a numerical type or a list type.")

    def to_string(self):
        """
        Finds and returns a string version of this `LinkedDistribution` of
        the form `"Linked(shared)"`, where `"shared"` is the string form of
        `LinkedDistribution.shared_distribution`.
        """
        return "Linked({!s})".format(self.shared_distribution.to_string())
    
    def __eq__(self, other):
        """
        Checks for equality of this `LinkedDistribution` with `other`.
        
        Parameters
        ----------
        other : object
            object to check for equality
        
        Returns
        -------
        result : bool
            True if and only if `other` is a `LinkedDistribution` with the same
            `LinkedDistribution.numparams` and
            `LinkedDistribution.shared_distribution`
        """
        if isinstance(other, LinkedDistribution):
            numparams_equal = (self.numparams == other.numparams)
            shared_distribution_equal =\
                (self.shared_distribution == other.shared_distribution)
            metadata_equal = self.metadata_equal(other)
            return all([numparams_equal, shared_distribution_equal,\
                metadata_equal])
        return False
    
    @property
    def minimum(self):
        """
        The minimum allowable value(s) in this distribution.
        """
        return [self.shared_distribution.minimum] * self.numparams
    
    @property
    def maximum(self):
        """
        The maximum allowable value(s) in this distribution.
        """
        return [self.shared_distribution.maximum] * self.numparams
    
    @property
    def is_discrete(self):
        """
        Boolean describing whether this distribution is discrete (True) or
        continuous (False).
        """
        return self.shared_distribution.is_discrete
    
    def fill_hdf5_group(self, group, save_metadata=True):
        """
        Fills the given hdf5 file group with data about this
        `LinkedDistribution` so that it can be loaded later.
        
        Parameters
        ----------
        group : h5py.Group
            hdf5 file group to fill
        save_metadata : bool
            - if True, attempts to save metadata alongside distribution and
            throws error if it fails
            - if False, metadata is ignored in saving process
        """
        group.attrs['class'] = 'LinkedDistribution'
        group.attrs['numparams'] = self.numparams
        subgroup = group.create_group('shared_distribution')
        self.shared_distribution.fill_hdf5_group(subgroup,\
            save_metadata=save_metadata)
        if save_metadata:
            self.save_metadata(group)
    
    @staticmethod
    def load_from_hdf5_group(group, shared_distribution_class, *args,\
        **kwargs):
        """
        Loads a `LinkedDistribution` from the given hdf5 file group.
        
        Parameters
        ----------
        group : h5py.Group
            the same hdf5 file group which fill_hdf5_group was called on when
            this Distribution was saved
        shared_distribution_class : class
            class of the univariate shared distribution
        args : sequence
            positional arguments to pass to the `load_from_hdf5_group` method
            of `shared_distribution_class`
        kwargs : dict
            keyword arguments to pass to the `load_from_hdf5_group` method of
            `shared_distribution_class`
        
        Returns
        -------
        distribution : `LinkedDistribution`
            distribution created from the information in the given group
        """
        try:
            assert group.attrs['class'] == 'LinkedDistribution'
        except:
            raise TypeError("The given hdf5 file doesn't seem to contain a " +\
                "LinkedDistribution.")
        metadata = Distribution.load_metadata(group)
        shared_distribution = shared_distribution_class.load_from_hdf5_group(\
            group['shared_distribution'], *args, **kwargs)
        numparams = group.attrs['numparams']
        return LinkedDistribution(shared_distribution, numparams,\
            metadata=metadata)
    
    @property
    def gradient_computable(self):
        """
        Boolean describing whether the gradient of the given distribution has
        been implemented. If True,
        `LinkedDistribution.gradient_of_log_value` method can be called
        safely.
        """
        return False 
    
    @property
    def hessian_computable(self):
        """
        Boolean describing whether the hessian of the given distribution has
        been implemented. If True,
        `LinkedDistribution.hessian_of_log_value` method can be called safely.
        """
        return False
    
    def copy(self):
        """
        Copies this distribution.
        
        Returns
        -------
        copied : `LinkedDistribution`
            a deep copy of this distribution, ignoring metadata.
        """
        return\
            LinkedDistribution(self.shared_distribution.copy(), self.numparams)

