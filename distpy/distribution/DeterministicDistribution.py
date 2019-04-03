"""
File: distpy/distribution/DeterministicDistribution.py
Author: Keith Tauscher
Date: 26 Jun 2018

Description: File containing a class for a pseudo-distribution, characterized
             by no values and deterministic drawn points.
"""
import numpy as np
from .Distribution import Distribution
from ..util import int_types, sequence_types, bool_types

class DeterministicDistribution(Distribution):
    """
    Class representing a deterministic distribution which simply yields draws
    which come from an array given at initialization.
    """
    def __init__(self, points, is_discrete=False, metadata=None):
        """
        Initializes a new DeterministicDistribution based around the given
        points.
        
        points: 1D (only if this distribution is 1D) or 2D array of points to
                return (in order)
        is_discrete: boolean describing whether the underlying distribution
                     from which points was sampled is discrete or continuous
        metadata: data to store alongside the distribution
        """
        self.points = points
        self.is_discrete = is_discrete
        self.metadata = metadata
    
    @property
    def points(self):
        """
        The points which will be returned (in order) by this deterministic
        distribution in a 2D array whose first dimension is the number of
        parameters of this distribution.
        """
        if not hasattr(self, '_points'):
            raise AttributeError("points was referenced before it was set.")
        return self._points
    
    @points.setter
    def points(self, value):
        """
        Setter for the points this distribution will return (in order).
        
        value: 1D (only if this distribution is 1D) or 2D array of points
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
        Property storing the maximum number of points this distribution can
        return (the same as the number of points originally given to this
        distribution).
        """
        if not hasattr(self, '_num_points'):
            self._num_points = self.points.shape[0]
        return self._num_points
    
    @property
    def current_index(self):
        """
        Property storing the index of the next point to return.
        """
        if not hasattr(self, '_current_index'):
            self._current_index = 0
        return self._current_index
    
    @current_index.setter
    def current_index(self, value):
        """
        Setter for the index of the next point to return.
        
        value: a positive integer
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
        Resets this DeterministicDistribution by setting the current_index to 0
        """
        if hasattr(self, '_current_index'):
            delattr(self, '_current_index')
    
    def draw(self, shape=None, random=None):
        """
        Draws a point from the distribution. Must be implemented by any base
        class.
        
        shape: if None, returns single random variate
                        (scalar for univariate ; 1D array for multivariate)
               if int, n, returns n random variates
                          (1D array for univariate ; 2D array for multivariate)
               if tuple of n ints, returns that many random variates
                                   n-D array for univariate ;
                                   (n+1)-D array for multivariate
        random: the random number generator to use (default: numpy.random)
        
        returns: either single value (if distribution is 1D) or array of values
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
                "DeterministicDistribution to return the desired shape. " +\
                "There are a total of {:d} points stored.").format(\
                len(self.points)))
    
    def log_value(self, point):
        """
        Computes the logarithm of the value of this distribution at the given
        point. It must be implemented by all subclasses.
        
        point: either single value (if distribution is 1D) or array of values
        
        returns: single number, logarithm of value of this distribution at the
                 given point
        """
        raise NotImplementedError("The DeterministicDistribution class can " +\
            "not be evaluated because it is not a real distribution. It " +\
            "can only be drawn from.")
    
    @property
    def gradient_computable(self):
        """
        Property which stores whether the gradient of the given distribution
        has been implemented.
        """
        return False
    
    @property
    def hessian_computable(self):
        """
        Property which stores whether the hessian of the given distribution
        has been implemented.
        """
        return False
    
    @property
    def numparams(self):
        """
        Property storing the integer number of parameters described by this
        distribution.
        """
        if self.points.ndim == 1:
            return 1
        else:
            return self.points.shape[1]
    
    def to_string(self):
        """
        Returns a string representation of this distribution. It must be
        implemented by all subclasses.
        """
        return "{:d}D DeterministicDistribution".format(self.numparams)
    
    def __eq__(self, other):
        """
        Tests for equality between this distribution and other. All subclasses
        must implement this function.
        
        other: Distribution with which to check for equality
        
        returns: True or False
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
        Property storing the minimum allowable value(s) in this distribution.
        """
        if not hasattr(self, '_minimum'):
            self._minimum = np.min(self.points, axis=0)
        return self._minimum
    
    @property
    def maximum(self):
        """
        Property storing the maximum allowable value(s) in this distribution.
        """
        if not hasattr(self, '_maximum'):
            self._maximum = np.max(self.points, axis=0)
        return self._maximum
    
    @property
    def is_discrete(self):
        """
        Property storing a boolean describing whether this distribution is
        discrete (True) or continuous (False).
        """
        if not hasattr(self, '_is_discrete'):
            raise AttributeError("is_discrete was referenced before it was " +\
                "set.")
        return self._is_discrete
    
    @is_discrete.setter
    def is_discrete(self, value):
        """
        Setter of whether the underlying distribution exists on a discrete
        domain.
        
        value: True or False
        """
        if type(value) in bool_types:
            self._is_discrete = value
        else:
            raise TypeError("is_discrete was set to a non-bool.")
    
    def fill_hdf5_group(self, group, save_metadata=True):
        """
        Fills the given hdf5 file group with information about this
        distribution. All subclasses must implement this function.
        
        group: hdf5 file group to fill with information about this distribution
        save_metadata: if True, attempts to save metadata alongside
                                distribution and throws error if it fails
                       if False, metadata is ignored in saving process
        """
        group.attrs['class'] = 'DeterministicDistribution'
        group.create_dataset('points', data=self.points)
        if save_metadata:
            self.save_metadata(group)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a Distribution from the given hdf5 file group. All Distribution
        subclasses must implement this method if things are to be saved in hdf5
        files.
        
        group: the same hdf5 file group which fill_hdf5_group was called on
               when this Distribution was saved
        
        returns: a Distribution object created from the information in the
                 given group
        """
        points = group['points'][()]
        metadata = Distribution.load_metadata(group)
        return DeterministicDistribution(points, metadata=metadata)
    
    def __mul__(self, other):
        """
        "Multiplies" this DeterministicDistribution by another
        DeterministicDistribution by combining them. The two distributions must
        contain the same number of points, but may contain different numbers of
        parameters. This function ignores metadata.
        
        other: another DeterministicDistribution to combine with this one
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
        Combines the given DeterministicDistributions into a single
        DeterministicDistribution. Same as product staticmethod of this class.
        
        distributions: the DeterministicDistribution objects to combine
        
        returns: DeterministicDistribution object which encodes the combination
                 of all of the given distributions
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
        Combines the given DeterministicDistributions into a single
        DeterministicDistribution. Should return same thing as combine
        staticmethod of this class but is implemented through repeated
        multiplication.
        
        distributions: the DeterministicDistribution objects to combine
        
        returns: DeterministicDistribution object which encodes the combination
                 of all of the given distributions
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
        Returns a deep copy of this Distribution. This function ignores
        metadata.
        """
        return DeterministicDistribution(self.points.copy())

