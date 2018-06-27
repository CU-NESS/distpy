"""
File: distpy/distribution/DeterministicDistribution.py
Author: Keith Tauscher
Date: 26 Jun 2018

Description: File containing a class for a pseudo-distribution, characterized
             by no values and deterministic drawn points.
"""
import numpy as np
from .Distribution import Distribution
from ..util import int_types, sequence_types

class DeterministicDistribution(Distribution):
    """
    
    """
    def __init__(self, points, metadata=None):
        """
        Initializes a new DeterministicDistribution based around the given
        points.
        
        points: 1D (only if this distribution is 1D) or 2D array of points to
                return (in order)
        metadata: data to store alongside the distribution
        """
        self.points = points
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
                raise 
        else:
            return 
    
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
        none_shape = (shape is None)
        if none_shape:
            shape = 1
        if type(shape) in int_types:
            shape = (shape,)
        total_number = np.prod(shape)
        if total_number <= (self.num_points - self.current_index):
            points_slice =\
                slice(self.current_index, self.current_index + self.num_points)
            to_return = self.points[points_slice]
            self.current_index = self.current_index + self.num_points
            if len(shape) == 1:
                if none_shape:
                    return to_return[0]
                else:
                    return to_return
            else:
                return np.reshape(to_return, shape)
        else:
            raise RuntimeError("Not enough points remain in this " +\
                "DeterministicDistribution to return the desired shape.")
    
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
    def is_discrete(self):
        """
        Property storing a boolean describing whether this distribution is
        discrete (True) or continuous (False).
        """
        return True
    
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
        points = group['points'].value
        metadata = Distribution.load_metadata(group)
        return DeterministicDistribution(points, metadata=metadata)

