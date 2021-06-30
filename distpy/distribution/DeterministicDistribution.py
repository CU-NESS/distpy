"""
Module containing class representing a "deterministic distribution," which is
essentially a sample, not a distribution. It is initialized with an array of
samples and can only be drawn from as many times as there are elements in that
array.

**File**: $DISTPY/distpy/distribution/DeterministicDistribution.py  
**Author**: Keith Tauscher  
**Date**: 31 May 2021
"""
import numpy as np
from .Distribution import Distribution
from ..util import int_types, sequence_types, bool_types, create_hdf5_dataset,\
    get_hdf5_value

class DeterministicDistribution(Distribution):
    """
    Class representing a "deterministic distribution," which is essentially a
    sample, not a distribution. It is initialized with an array of samples and
    can only be drawn from as many times as there are elements in that array.
    """
    def __init__(self, points, is_discrete=False, metadata=None):
        """
        Initializes a new `DeterministicDistribution` with the given parameter
        values.
        
        Parameters
        ----------
        points : `numpy.ndarray`
            the sequence of points in an array of shape \\(N\\) if this is a
            univariate distribution or \\((N,n)\\) if it is an
            \\(N\\)-dimensional distribution (\\(n\\) is arbitrary)
        is_discrete : bool
            bool determining whether this `DeterministicDistribution` should be
            considered discrete
        metadata : number or str or dict or `distpy.util.Savable.Savable`
            data to store alongside this distribution.
        """
        self.points = points
        self.is_discrete = is_discrete
        self.metadata = metadata
    
    @property
    def points(self):
        """
        The points which will be returned (in order) by this
        `DeterministicDistribution` in a 2D array whose first dimension is the
        number of parameters of this distribution.
        """
        if not hasattr(self, '_points'):
            raise AttributeError("points was referenced before it was set.")
        return self._points
    
    @points.setter
    def points(self, value):
        """
        Setter for `DeterministicDistribution.points`.
        
        Parameters
        ----------
        value : numpy.ndarray
            - if this distribution is 1D, `value` should be a 1D numpy array of
            floats
            - if this distribution is \\(n\\)-dimensional, `value` should be a
            2D numpy array whose second axis has length \\(n\\)
        """
        if type(value) in sequence_types:
            value = np.array(value)
            if value.ndim == 1:
                value = value[:,np.newaxis]
            if value.ndim != 2:
                raise ValueError("points was set to an array whose number " +\
                    "of dimensions was not 1 or 2.")
            if value.shape[-1] == 1:
                self._points = value[:,0]
            else:
                self._points = value
        else:
            raise TypeError("points was set to a non-sequence.")
    
    @property
    def num_points(self):
        """
        The integer maximum number of points this distribution can return (the
        same as the number of points originally given to this distribution).
        """
        if not hasattr(self, '_num_points'):
            self._num_points = self.points.shape[0]
        return self._num_points
    
    @property
    def current_index(self):
        """
        The integer index of the next point to return.
        """
        if not hasattr(self, '_current_index'):
            self._current_index = 0
        return self._current_index
    
    @current_index.setter
    def current_index(self, value):
        """
        Setter for `DeterministicDistribution.current_index`.
        
        Parameters
        ----------
        value : int
            integer index of the next point to return
        """
        if type(value) in int_types:
            if value >= 0:
                self._current_index = value
            else:
                raise ValueError("current_index was set to a negative number.")
        else:
            raise TypeError("current index was set to a non-integer.")
    
    def reset(self):
        """
        Resets this `DeterministicDistribution` by setting
        `DeterministicDistribution.current_index` to 0
        """
        if hasattr(self, '_current_index'):
            delattr(self, '_current_index')
    
    def draw(self, shape=None, random=None):
        """
        Draws point(s) from this `DeterministicDistribution`.
        
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
        none_shape = (type(shape) is type(None))
        if none_shape:
            shape = 1
        if type(shape) in int_types:
            shape = (shape,)
        total_number = np.prod(shape)
        if total_number <= (self.num_points - self.current_index):
            points_slice =\
                slice(self.current_index, self.current_index + total_number)
            to_return = self.points[points_slice]
            self.current_index = self.current_index + total_number
            if len(shape) == 1:
                if none_shape:
                    return to_return[0]
                else:
                    return to_return
            else:
                return np.reshape(to_return, shape)
        else:
            raise RuntimeError(("Not enough points remain in this " +\
                "DeterministicDistribution to return the desired shape, " +\
                "which is {0}. There are a total of {1:d} points " +\
                "stored and there are {2:d} remaining.").format(shape,\
                self.num_points, self.num_points - self.current_index))
    
    def log_value(self, point):
        """
        The `DeterministicDistribution` is improper, so its log value cannot be
        computed.
        """
        raise NotImplementedError("The DeterministicDistribution class can " +\
            "not be evaluated because it is not a real distribution. It " +\
            "can only be drawn from.")
    
    @property
    def gradient_computable(self):
        """
        Boolean describing whether the gradient of the given distribution has
        been implemented. If True,
        `DeterministicDistribution.gradient_of_log_value` method can be called
        safely.
        """
        return False
    
    @property
    def hessian_computable(self):
        """
        Boolean describing whether the hessian of the given distribution has
        been implemented. If True,
        `DeterministicDistribution.hessian_of_log_value` method can be called
        safely.
        """
        return False
    
    @property
    def numparams(self):
        """
        The number of parameters of this `DiscreteDistribution`.
        """
        if self.points.ndim == 1:
            return 1
        else:
            return self.points.shape[1]
    
    @property
    def mean(self):
        """
        The mean of this `DeterministicDistribution`, which is the mean of the
        points that it returns.
        """
        if not hasattr(self, '_mean'):
            self._mean = np.mean(self.points, axis=0)
        return self._mean
    
    @property
    def variance(self):
        """
        The (co)variance of this `DeterministicDistribution`.
        """
        if not hasattr(self, '_variance'):
            if self.numparams == 1:
                self._variance = np.var(self.points)
            else:
                self._variance = np.cov(self.points, rowvar=False)
        return self._variance
    
    def to_string(self):
        """
        Finds and returns a string version of this `DeterministicDistribution`
        of the form `"nD DeterministicDistribution"`.
        """
        return "{:d}D DeterministicDistribution".format(self.numparams)
    
    def __eq__(self, other):
        """
        Checks for equality of this `DeterministicDistribution` with `other`.
        
        Parameters
        ----------
        other : object
            object to check for equality
        
        Returns
        -------
        result : bool
            True if and only if `other` is a `DeterministicDistribution` with
            the same `DeterministicDistribution.points`
        """
        if isinstance(other, DeterministicDistribution):
            if self.points.shape == other.points.shape:
                metadata_equal = self.metadata_equal(other)
                points_equal = np.allclose(self.points, other.points)
                return (metadata_equal and points_equal)
            else:
                return False
        else:
            return False
    
    @property
    def minimum(self):
        """
        The minimum allowable value(s) in this distribution.
        """
        if not hasattr(self, '_minimum'):
            self._minimum = np.min(self.points, axis=0)
        return self._minimum
    
    @property
    def maximum(self):
        """
        The maximum allowable value(s) in this distribution.
        """
        if not hasattr(self, '_maximum'):
            self._maximum = np.max(self.points, axis=0)
        return self._maximum
    
    @property
    def is_discrete(self):
        """
        Boolean describing whether this distribution is discrete (True) or
        continuous (False).
        """
        if not hasattr(self, '_is_discrete'):
            raise AttributeError("is_discrete was referenced before it was " +\
                "set.")
        return self._is_discrete
    
    @is_discrete.setter
    def is_discrete(self, value):
        """
        Setter for `DeterministicDistribution.is_discrete`.
        
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
        `DeterministicDistribution` so that it can be loaded later.
        
        Parameters
        ----------
        group : h5py.Group
            hdf5 file group to fill
        save_metadata : bool
            - if True, attempts to save metadata alongside distribution and
            throws error if it fails
            - if False, metadata is ignored in saving process
        """
        group.attrs['class'] = 'DeterministicDistribution'
        create_hdf5_dataset(group, 'points', data=self.points)
        if save_metadata:
            self.save_metadata(group)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a `DeterministicDistribution` from the given hdf5 file group.
        
        Parameters
        ----------
        group : h5py.Group
            the same hdf5 file group which fill_hdf5_group was called on when
            this Distribution was saved
        
        Returns
        -------
        distribution : `DeterministicDistribution`
            distribution created from the information in the given group
        """
        points = get_hdf5_value(group['points'])
        metadata = Distribution.load_metadata(group)
        return DeterministicDistribution(points, metadata=metadata)
    
    def __mul__(self, other):
        """
        "Multiplies" this `DeterministicDistribution` by another
        `DeterministicDistribution` by combining them. The two distributions
        must contain the same number of points, but may contain different
        numbers of parameters. This function ignores metadata.
        
        Parameters
        ----------
        other : `DeterministicDistribution`
            another `DeterministicDistribution` to combine with this one
        
        Returns
        -------
        combined : `DeterministicDistribution`
            if this distribution represented the distribution of
            \\(\\boldsymbol{x}\\) and `other` represented the distribution of
            \\(\\boldsymbol{y}\\), `combined` represents the distribution of
            \\(\\begin{bmatrix} \\boldsymbol{x} \\\\ \\boldsymbol{y}\
            \\end{bmatrix}\\)
        """
        if isinstance(other, DeterministicDistribution):
            self_points_slice = (None if self.numparams == 1 else slice(None))
            other_points_slice =\
                (None if other.numparams == 1 else slice(None))
            new_points = np.concatenate([self.points[:,self_points_slice],\
                other.points[:,other_points_slice]], axis=-1)
            return DeterministicDistribution(new_points)
        else:
            raise TypeError("A DeterministicDistribution can only be " +\
                "combined to other DeterministicDistribution.")
    
    @staticmethod
    def combine(*distributions):
        """
        Combines many `DeterministicDistribution` objects. The distributions
        must contain the same number of points, but may contain different
        numbers of parameters. This function ignores metadata. Same as
        `DeterministicDistribution.product` static method.
        
        Parameters
        ----------
        distributions : sequence
            sequence of `DeterministicDistribution` objects to combine
        
        Returns
        -------
        combined : `DeterministicDistribution`
            if `distributions` represented the distributions of
            \\(\\boldsymbol{x}_1,\\boldsymbol{x}_2,\\ldots,\
            \\boldsymbol{x}_N\\), then `combined` represents the distribution
            of \\(\\begin{bmatrix} \\boldsymbol{x}_1 \\\\\
            \\boldsymbol{x}_2 \\\\ \\vdots \\\\ \\boldsymbol{x}_N\
            \\end{bmatrix}\\)
        """
        if all([isinstance(distribution, DeterministicDistribution)\
            for distribution in distributions]):
            num_points =\
                [distribution.num_points for distribution in distributions]
            if all([(npoints == num_points[0]) for npoints in num_points]):
                slices = [None if distribution.numparams == 1 else slice(None)\
                    for distribution in distributions]
                new_points = np.concatenate([distribution.points[:,slc]\
                    for (distribution, slc) in zip(distributions, slices)],\
                    axis=1)
                return DeterministicDistribution(new_points)
            else:
                raise ValueError("Can only combine " +\
                    "DeterministicDistributions which have the same " +\
                    "num_points property.")
        else:
            raise TypeError("Can only combine multiple " +\
                "DeterministicDistributions.")
    
    @staticmethod
    def product(*distributions):
        """
        Combines many `DeterministicDistribution` objects. The distributions
        must contain the same number of points, but may contain different
        numbers of parameters. This function ignores metadata.
        
        Parameters
        ----------
        distributions : sequence
            sequence of `DeterministicDistribution` objects to combine
        
        Returns
        -------
        combined : `DeterministicDistribution`
            if `distributions` represented the distributions of
            \\(\\boldsymbol{x}_1,\\boldsymbol{x}_2,\\ldots,\
            \\boldsymbol{x}_N\\), then `combined` represents the distribution
            of \\(\\begin{bmatrix} \\boldsymbol{x}_1 \\\\\
            \\boldsymbol{x}_2 \\\\ \\vdots \\\\ \\boldsymbol{x}_N\
            \\end{bmatrix}\\)
        """
        product_distribution = None
        for distribution in distributions:
            if type(product_distribution) is type(None):
                product_distribution = distribution
            else:
                product_distribution = product_distribution * distribution
        return product_distribution
    
    def copy(self):
        """
        Copies this distribution.
        
        Returns
        -------
        copied : `DeterministicDistribution`
            a deep copy of this distribution, ignoring metadata.
        """
        return DeterministicDistribution(self.points.copy())

