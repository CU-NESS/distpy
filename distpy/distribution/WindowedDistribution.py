"""
Module containing class representing a so-called windowed distribution, which
follows a distribution (the so-called background distribution) with a support
given by that of a different distribution (the so-called foreground
distribution). Its PDF is represented by:
$$f(x)\\propto \\begin{cases} g(x) & h(x)\\ne 0 \\\\ 0 & h(x)=0 \\end{cases},$$
where \\(g\\) is the background distribution PDF and \\(h\\) is the foreground
distribution PDF.

**File**: $DISTPY/distpy/distribution/WindowedDistribution.py  
**Author**: Keith Tauscher  
**Date**: 31 May 2021
"""
from __future__ import division
import numpy as np
import numpy.random as rand
from ..util import int_types
from .Distribution import Distribution

class WindowedDistribution(Distribution):
    """
    Class representing a so-called windowed distribution, which follows a
    distribution (the so-called background distribution) with a support given
    by that of a different distribution (the so-called foreground
    distribution). Its PDF is represented by: $$f(x)\\propto \\begin{cases}\
    g(x) & h(x)\\ne 0 \\\\ 0 & h(x)=0 \\end{cases},$$ where \\(g\\) is the
    background distribution PDF and \\(h\\) is the foreground distribution PDF.
    """
    def __init__(self, background_distribution, foreground_distribution,\
        metadata=None):
        """
        Initializes a new `WindowedDistribution` with the given parameter
        values.
        
        Parameters
        ----------
        background_distribution : `distpy.distribution.Distribution.Distribution`
            distribution with PDF \\(g\\) that is drawn from
        foreground_distribution : `distpy.distribution.Distribution.Distribution`
            distribution with PDF \\(h\\) that is used to determine whether a
            point should be accepted or rejected
        metadata : number or str or dict or `distpy.util.Savable.Savable`
            data to store alongside this distribution.
        """
        self.background_distribution = background_distribution
        self.foreground_distribution = foreground_distribution
        self.metadata = metadata
    
    @property
    def background_distribution(self):
        """
        The distribution from which points are drawn.
        """
        if not hasattr(self, '_background_distribution'):
            raise AttributeError("background_distribution was referenced " +\
                "before it was set.")
        return self._background_distribution
    
    @background_distribution.setter
    def background_distribution(self, value):
        """
        Setter for `WindowedDistribution.background_distribution`.
        
        Parameters
        ----------
        value : `distpy.distribution.Distribution.Distribution`
            background distribution to draw from
        """
        if isinstance(value, Distribution):
            self._background_distribution = value
        else:
            raise TypeError("background_distribution was set to something " +\
                "other than a Distribution object.")
    
    @property
    def foreground_distribution(self):
        """
        The distribution which forms the window.
        """
        if not hasattr(self, '_foreground_distribution'):
            raise AttributeError("foreground_distribution was referenced " +\
                "before it was set.")
        return self._foreground_distribution
    
    @foreground_distribution.setter
    def foreground_distribution(self, value):
        """
        Setter for `WindowedDistribution.foreground_distribution`.
        
        Parameters
        ----------
        value : `distpy.distribution.Distribution.Distribution`
            foreground distribution to reject points with
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
        The number of parameters of this `WindowedDistribution`, which is the
        number of parameters of `WindowedDistribution.background_distribution`
        and `WindowedDistribution.foreground_distribution`.
        """
        return self.background_distribution.numparams
    
    @property
    def mean(self):
        """
        The mean of the `WindowedDistribution` class is not implemented.
        """
        if not hasattr(self, '_mean'):
            raise NotImplementedError("mean is not implemented for the " +\
                "WindowedDistribution class.")
        return self._mean
    
    @property
    def variance(self):
        """
        The (co)variance of the `WindowedDistribution` class is not
        implemented.
        """
        if not hasattr(self, '_variance'):
            raise AttributeError("variance is not implemented for the " +\
                "WindowedDistribution class.")
        return self._variance
    
    def draw(self, shape=None, random=rand):
        """
        Draws point(s) from this `WindowedDistribution`.
        
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
        Approximates the fraction of draws from this `WindowedDistribution`
        object's `WindowedDistribution.background_distribution` that are
        accepted by the `WindowedDistribution.foreground_distribution`. In the
        ideal case (i.e. when `num_to_draw` is, impossibly, infinite), this is
        equal to the fraction of probability of the
        `WindowedDistribution.background_distribution` which has a log value of
        0 when evaluated with `WindowedDistribution.foreground_distribution`.
        
        Parameters
        ----------
        num_to_draw : int
            integer number of draws to make when approximating. The larger this
            number the better the approximation.
        
        Returns
        -------
        acceptance_fraction : float
            single number between 0 and 1 which is a multiple of
            1/`num_to_draw`
        """
        draws = self.background_distribution.draw(num_to_draw)
        foreground_log_values =\
            [self.foreground_distribution.log_value(draw) for draw in draws]
        return np.sum(np.isfinite(foreground_log_values)) / num_to_draw
    
    def approximate_rejection_fraction(self, num_to_draw):
        """
        Approximates the fraction of draws from this `WindowedDistribution`
        object's `WindowedDistribution.background_distribution` that are
        rejected by the `WindowedDistribution.foreground_distribution`. In the
        ideal case (i.e. when `num_to_draw` is, impossibly, infinite), this is
        equal to one minus the fraction of probability of the
        `WindowedDistribution.background_distribution` which has a log value of
        0 when evaluated with `WindowedDistribution.foreground_distribution`.
        
        Parameters
        ----------
        num_to_draw : int
            integer number of draws to make when approximating. The larger this
            number the better the approximation.
        
        Returns
        -------
        rejection_fraction : float
            single number between 0 and 1 which is a multiple of
            1/`num_to_draw`
        """
        return 1 - self.approximate_acceptance_fraction(num_to_draw)

    def log_value(self, point):
        """
        Computes the logarithm of the value of this `WindowedDistribution` at
        the given point (which may be off by a `point`-independent constant).
        
        Parameters
        ----------
        point : float or `numpy.ndarray`
            - if this distribution is univariate, `point` should be a scalar
            - if this distribution describes \\(p\\) parameters, `point` should
            be a length-\\(p\\) `numpy.ndarray`
        
        Returns
        -------
        value : float
            natural logarithm of the value of this distribution at `point`. If
            \\(f\\) is this distribution's PDF and \\(x\\) is `point`, then
            `value` is \\(\\ln{\\big(f(x)\\big)}\\)
        """
        if np.isfinite(self.foreground_distribution.log_value(point)):
            return self.background_distribution.log_value(point)
        else:
            return (-np.inf)

    def to_string(self):
        """
        Finds and returns a string version of this `WindowedDistribution` of
        the form `"Windowed(background, foreground)"`, where `"background"` and
        `"foreground"` are the string representations of
        `WindowedDistribution.background_distribution` and
        `WindowedDistribution.foreground_distribution`, respectively.
        """
        return "Windowed({0!s},{1!s})".format(\
            self.background_distribution.to_string(),\
            self.foreground_distribution.to_string())
    
    def __eq__(self, other):
        """
        Checks for equality of this `WindowedDistribution` with `other`.
        
        Parameters
        ----------
        other : object
            object to check for equality
        
        Returns
        -------
        result : bool
            True if and only if `other` is a `WindowedDistribution` with the
            same `WindowedDistribution.background_distribution` and
            `WindowedDistribution.foreground_distribution`
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
        The minimum allowable value(s) in this distribution.
        """
        if not hasattr(self, '_minimum'):
            minimum = []
            background_minimum = self.background_distribution.minimum
            foreground_minimum = self.foreground_distribution.minimum
            if self.numparams == 1:
                background_minimum = [background_minimum]
                foreground_minimum = [foreground_minimum]
            for index in range(self.numparams):
                if type(background_minimum[index]) is type(None):
                    minimum.append(foreground_minimum[index])
                elif type(foreground_minimum[index]) is type(None):
                    minimum.append(background_minimum[index])
                else:
                    minimum.append(max(\
                        foreground_minimum[index], background_minimum[index]))
            if self.numparams == 1:
                minimum = minimum[0]
            self._minimum = minimum
        return self._minimum
    
    @property
    def maximum(self):
        """
        The maximum allowable value(s) in this distribution.
        """
        if not hasattr(self, '_maximum'):
            maximum = []
            background_maximum = self.background_distribution.maximum
            foreground_maximum = self.foreground_distribution.maximum
            for index in range(self.numparams):
                if type(background_maximum[index]) is type(None):
                    maximum.append(foreground_maximum[index])
                elif type(foreground_maximum[index]) is type(None):
                    maximum.append(background_maximum[index])
                else:
                    maximum.append(min(\
                        foreground_maximum[index], background_maximum[index]))
            if self.numparams == 1:
                maximum = maximum[0]
            self._maximum = maximum
        return self._maximum
    
    @property
    def is_discrete(self):
        """
        Boolean describing whether this distribution is discrete (True) or
        continuous (False).
        """
        return self.background_distribution.is_discrete
    
    def fill_hdf5_group(self, group, save_metadata=True):
        """
        Fills the given hdf5 file group with data about this
        `WindowedDistribution` so that it can be loaded later.
        
        Parameters
        ----------
        group : h5py.Group
            hdf5 file group to fill
        save_metadata : bool
            - if True, attempts to save metadata alongside distribution and
            throws error if it fails
            - if False, metadata is ignored in saving process
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
        Loads a `WindowedDistribution` from the given hdf5 file group.
        
        Parameters
        ----------
        group : h5py.Group
            the same hdf5 file group which fill_hdf5_group was called on when
            this Distribution was saved
        background_distribution_class : class
            the `distpy.distribution.Distribution.Distribution` subclass which
            should be loaded from this group as background distribution
        foreground_distribution_class : class
            the `distpy.distribution.Distribution.Distribution` subclass which
            should be loaded from this group as foreground distribution
        args : sequence
            positional arguments to pass on to `load_from_hdf5_group` method of
            `foreground_distribution_class`
        kwargs : dict
            keyword arguments to pass on to `load_from_hdf5_group` method of
            `foreground_distribution_class`
        
        Returns
        -------
        distribution : `WindowedDistribution`
            distribution created from the information in the given group
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
        Boolean describing whether the gradient of the given distribution has
        been implemented. If True,
        `WindowedDistribution.gradient_of_log_value` method can be called
        safely.
        """
        return self.background_distribution.gradient_computable
    
    def gradient_of_log_value(self, point):
        """
        Computes the gradient (derivative) of the logarithm of the value of
        this `WindowedDistribution` at the given point.
        
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
        return self.background_distribution.gradient_of_log_value(point)
    
    @property
    def hessian_computable(self):
        """
        Boolean describing whether the hessian of the given distribution has
        been implemented. If True,
        `WindowedDistribution.hessian_of_log_value` method can be called
        safely.
        """
        return self.background_distribution.hessian_computable
    
    def hessian_of_log_value(self, point):
        """
        Computes the hessian (second derivative) of the logarithm of the value
        of this `WindowedDistribution` at the given point.
        
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
        return self.background_distribution.hessian_of_log_value(point)
    
    def copy(self):
        """
        Copies this distribution.
        
        Returns
        -------
        copied : `WindowedDistribution`
            a deep copy of this distribution, ignoring metadata.
        """
        return WindowedDistribution(self.background_distribution.copy(),\
            self.foreground_distribution.copy())

