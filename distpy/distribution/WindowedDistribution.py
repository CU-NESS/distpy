"""
File: distpy/distribution/WindowedDistribution.py
Author: Keith Tauscher
Date: 23 Sep 2018

Description: File containing distribution which is similar to a given one but,
             when drawing from them, only points which exist where a different
             distribution is nonzero.
"""
from __future__ import division
import numpy as np
from ..util import int_types
from .Distribution import Distribution

class WindowedDistribution(Distribution):
    """
    Class representing a distribution which is similar to a given one but,
    when drawing from them, only points which exist where a different
    distribution is nonzero.
    """
    def __init__(self, background_distribution, foreground_distribution,\
        metadata=None):
        """
        Initializes a new WindowedDistribution with the given underlying
        distribution and window distribution.
        
        background_distribution: distribution from which points are drawn
        foreground_distribution: distribution with which to window the other
                                 distribution
        metadata: store alongside the distribution
        """
        self.background_distribution = background_distribution
        self.foreground_distribution = foreground_distribution
        self.metadata = metadata
    
    @property
    def background_distribution(self):
        """
        Property storing the distribution from which points are drawn.
        """
        if not hasattr(self, '_background_distribution'):
            raise AttributeError("background_distribution was referenced " +\
                "before it was set.")
        return self._background_distribution
    
    @background_distribution.setter
    def background_distribution(self, value):
        """
        Setter for the background distribution.
        
        value: Distribution object
        """
        if isinstance(value, Distribution):
            self._background_distribution = value
        else:
            raise TypeError("background_distribution was set to something " +\
                "other than a Distribution object.")
    
    @property
    def foreground_distribution(self):
        """
        Property storing the distribution which forms the window.
        """
        if not hasattr(self, '_foreground_distribution'):
            raise AttributeError("foreground_distribution was referenced " +\
                "before it was set.")
        return self._foreground_distribution
    
    @foreground_distribution.setter
    def foreground_distribution(self, value):
        """
        Setter for the foreground distribution.
        
        value: Distribution object with the same number of parameters as
               background distribution
        """
        if isinstance(value, Distribution):
            if value.numparams == self.background_distribution.numparams:
                self._foreground_distribution = value
            else:
                raise ValueError("foreground_distribution and " +\
                    "background_distribution do not have the same number " +\
                    "of parameters.")
        else:
            raise TypeError("foreground_distribution was set to something " +\
                "other than a Distribution object.")
    
    @property
    def numparams(self):
        """
        Finds and returns the number of parameters which are described by this
        WindowedDistribution.
        """
        return self.background_distribution.numparams
    
    def draw(self, shape=None, random=np.random):
        """
        Draws values from background_distribution, replacing points which are
        drawn in places of zero probability under foreground_distribution.
        
        shape: if None, returns single random variate
                        (scalar for univariate ; 1D array for multivariate)
               if int, n, returns n random variates
                          (1D array for univariate ; 2D array for multivariate)
               if tuple of n ints, returns that many random variates
                                   n-D array for univariate ;
                                   (n+1)-D array for multivariate
        random: the random number generator to use (default: numpy.random)
        
        returns numpy.ndarray of values (sorted by design)
        """
        none_shape = (shape is None)
        if none_shape:
            shape = 1
        if type(shape) in int_types:
            shape = (shape,)
        num_to_draw = np.prod(shape)
        if self.numparams == 1:
            points = np.ndarray((0,))
        else:
            points = np.ndarray((0, self.numparams))
        while points.shape[0] < num_to_draw:
            remaining_to_draw = num_to_draw - points.shape[0]
            new_draws = self.background_distribution.draw(remaining_to_draw,\
                random=random)
            foreground_log_values =\
                [self.foreground_distribution.log_value(draw)\
                for draw in new_draws]
            new_draws = new_draws[np.isfinite(foreground_log_values),...]
            points = np.concatenate([points, new_draws], axis=0)
        points = points[:num_to_draw,...]
        if len(shape) != 1:
            points = np.reshape(points, shape + points.shape[-1:])
        if none_shape:
            return points[0]
        else:
            return points
    
    def approximate_acceptance_fraction(self, num_to_draw):
        """
        Approximates the fraction of draws from this Distribution's
        background_distribution that are accepted by the
        foreground_distribution. In the ideal case (i.e. when num_to_draw is,
        impossibly, infinite), this is equal to the fraction of probability of
        the background_distribution which has a log value of 0 when
        evaluated with the foreground_distribution.
        
        num_to_draw: integer number of draws to make when approximating. The
                     larger this number the better the approximation.
        
        returns: single number between 0 and 1 which is a multiple of
                 1/num_to_draw
        """
        draws = self.background_distribution.draw(num_to_draw)
        foreground_log_values =\
            [self.foreground_distribution.log_value(draw) for draw in draws]
        return np.sum(np.isfinite(foreground_log_values)) / num_to_draw
    
    def approximate_rejection_fraction(self, num_to_draw):
        """
        Approximates the fraction of draws from this Distribution's
        background_distribution that are rejected by the
        foreground_distribution. In the ideal case (i.e. when num_to_draw is,
        impossibly, infinite), this is equal to the fraction of probability of
        the background_distribution which has a log value of -np.inf when
        evaluated with the foreground_distribution.
        
        num_to_draw: integer number of draws to make when approximating. The
                     larger this number the better the approximation.
        
        returns: single number between 0 and 1 which is a multiple of
                 1/num_to_draw
        """
        return 1 - self.approximate_acceptance_fraction(num_to_draw)

    def log_value(self, point):
        """
        Evaluates and returns the log_value at the given point. For this
        distribution, the log value may be off by a parameter independent
        additive constant.
        """
        if np.isfinite(self.foreground_distribution.log_value(point)):
            return self.background_distribution.log_value(point)
        else:
            return (-np.inf)

    def to_string(self):
        """
        Finds and returns a string representation of this
        WindowedDistribution.
        """
        return "Windowed({0!s},{1!s})".format(\
            self.background_distribution.to_string(),\
            self.foreground_distribution.to_string())
    
    def __eq__(self, other):
        """
        Checks for equality of this distribution with other. Returns True if
        other is a WindowedDistribution with the same number of parameters and
        background and foreground distributions and False otherwise.
        """
        if isinstance(other, WindowedDistribution):
            if self.numparams == other.numparams:
                return ((self.background_distribution ==\
                    other.background_distribution) and\
                    (self.foreground_distribution ==\
                    other.foreground_distribution))
            else:
                return False
        else:
            return False
    
    @property
    def minimum(self):
        """
        Property storing the minimum allowable value(s) in this distribution.
        """
        return np.where(self.foreground_distribution.minimum >\
            self.background_distribution.minimum,\
            self.foreground_distribution.minimum,\
            self.background_distribution.minimum)
    
    @property
    def maximum(self):
        """
        Property storing the maximum allowable value(s) in this distribution.
        """
        return np.where(self.foreground_distribution.maximum <\
            self.background_distribution.maximum,\
            self.foreground_distribution.maximum,\
            self.background_distribution.maximum)
    
    @property
    def is_discrete(self):
        """
        Property storing a boolean describing whether this distribution is
        discrete (True) or continuous (False).
        """
        return self.background_distribution.is_discrete
    
    def fill_hdf5_group(self, group, save_metadata=True):
        """
        Fills the given hdf5 file group with data from this distribution. That
        data includes the class name, the number of parameters, and the shared
        distribution.
        
        group: hdf5 file group to fill
        save_metadata: if True, attempts to save metadata alongside
                                distribution and throws error if it fails
                       if False, metadata is ignored in saving process
        """
        group.attrs['class'] = 'WindowedDistribution'
        subgroup = group.create_group('background_distribution')
        self.background_distribution.fill_hdf5_group(subgroup)
        subgroup = group.create_group('foreground_distribution')
        self.foreground_distribution.fill_hdf5_group(subgroup)
        if save_metadata:
            self.save_metadata(group)
    
    @staticmethod
    def load_from_hdf5_group(group, background_distribution_class,\
        foreground_distribution_class, *args, **kwargs):
        """
        Loads a WindowedDistribution from the given hdf5 file group.
        
        group: the same hdf5 file group which fill_hdf5_group was called on
               when this Distribution was saved
        background_distribution_class: the Distribution subclass which should
                                       be loaded from this group as background
                                       distribution
        foreground_distribution_class: the Distribution subclass which should
                                       be loaded from this group as foreground
                                       distribution
        args: positional arguments to pass on to load_from_hdf5_group method of
              foreground_distribution_class
        kwargs: keyword arguments to pass on to load_from_hdf5_group method of
                foreground_distribution_class
        
        returns: a WindowedDistribution object created from the information in
                 the given group
        """
        try:
            assert group.attrs['class'] == 'WindowedDistribution'
        except:
            raise TypeError("The given hdf5 file doesn't seem to contain a " +\
                "WindowedDistribution.")
        metadata = Distribution.load_metadata(group)
        background_distribution =\
            background_distribution_class.load_from_hdf5_group(\
            group['background_distribution'])
        foreground_distribution =\
            foreground_distribution_class.load_from_hdf5_group(\
            group['foreground_distribution'], *args, **kwargs)
        return WindowedDistribution(background_distribution,\
            foreground_distribution, metadata=metadata)
    
    @property
    def gradient_computable(self):
        """
        Property which stores whether the gradient of the given distribution
        has been implemented. Since it has been implemented, it returns True.
        """
        return self.background_distribution.gradient_computable
    
    def gradient_of_log_value(self, point):
        """
        Computes the derivative of log_value(point) with respect to the
        parameter.
        
        point: vector of values at which to evaluate derivatives
        
        returns: returns single number representing derivative of log value
        """
        return self.background_distribution.gradient_of_log_value(point)
    
    @property
    def hessian_computable(self):
        """
        Property which stores whether the hessian of the given distribution
        has been implemented. Since it has been implemented, it returns True.
        """
        return self.background_distribution.hessian_computable
    
    def hessian_of_log_value(self, point):
        """
        Computes the second derivative of log_value(point) with respect to the
        parameter.
        
        point: vector of values at which to evaluate second derivatives
        
        returns: single number representing second derivative of log value
        """
        return self.background_distribution.hessian_of_log_value(point)
    
    def copy(self):
        """
        Returns a deep copy of this Distribution. This function ignores
        metadata.
        """
        return WindowedDistribution(background_distribution.copy(),\
            foreground_distribution.copy())

