"""
Module containing class representing a jumping distribution which describes a
translation whose distribution does not depend on the starting location. If the
PDF of the translation distribution is \\(g(\\boldsymbol{t})\\), this
distribution's PDF is given by
$$f(\\boldsymbol{x},\\boldsymbol{y})=g(\\boldsymbol{y}-\\boldsymbol{x})$$

**File**: $DISTPY/distpy/jumping/LocalIndependentJumpingDistribution.py  
**Author**: Keith Tauscher  
**Date**: 11 Jul 2021
"""
import numpy as np
from ..util import int_types
from ..distribution import Distribution, load_distribution_from_hdf5_group
from .JumpingDistribution import JumpingDistribution

class LocaleIndependentJumpingDistribution(JumpingDistribution):
    """
    Class representing a jumping distribution which describes a translation
    whose distribution does not depend on the starting location. If the PDF of
    the translation distribution is \\(g(\\boldsymbol{t})\\), this
    distribution's PDF is given by
    $$f(\\boldsymbol{x},\\boldsymbol{y})=g(\\boldsymbol{y}-\\boldsymbol{x})$$
    """
    def __init__(self, distribution):
        """
        Initializes this `LocaleIndependentJumpingDistribution` with the given
        core distribution.
        
        Parameters
        ----------
        distribution : `distpy.distribution.Distribution.Distribution`
            a distribution describing the probability of the displacement
            between the source and destination.
        """
        self.distribution = distribution
    
    @property
    def distribution(self):
        """
        The distribution describing the probability of the displacement between
        the source and destination.
        """
        if not hasattr(self, '_distribution'):
            raise AttributeError("distribution referenced before it was set.")
        return self._distribution
    
    @distribution.setter
    def distribution(self, value):
        """
        Setter for `LocaleIndependentJumpingDistribution.distribution`
        
        Parameters
        ----------
        value : `distpy.distribution.Distribution.Distribution`
            a distribution describing the probability of the displacement
            between the source and destination.
        """
        if isinstance(value, Distribution):
            self._distribution = value
        else:
            raise TypeError("distribution was not a Distribution object.")
    
    def draw(self, source, shape=None, random=np.random):
        """
        Draws a destination point from this jumping distribution given a source
        point.
        
        Parameters
        ----------
        source : number or numpy.ndarray
            - if this `LocaleIndependentJumpingDistribution` is univariate,
            source should be a single number
            - otherwise, source should be `numpy.ndarray` of shape (numparams,)
        shape : None or int or tuple
            - if None, a single destination is returned
                - if this distribution is univariate, a single number is
                returned
                - if this distribution is multivariate, a 1D `numpy.ndarray`
                describing the coordinates of the destination is returned
            - if int \\(n\\), \\(n\\) destinations are returned
                - if this distribution is univariate, a 1D `numpy.ndarray` of
                length \\(n\\) is returned
                - if this distribution describes \\(p\\) dimensions, a 2D
                `numpy.ndarray` is returned whose shape is \\((n,p)\\)
            - if tuple of ints \\((n_1,n_2,\\ldots,n_k)\\),
            \\(\\prod_{m=1}^kn_m\\) destinations are returned
                - if this distribution is univariate, a `numpy.ndarray` of
                shape \\((n_1,n_2,\\ldots,n_k)\\) is returned
                - if this distribution describes \\(p\\) parameters, a
                `numpy.ndarray` of shape \\((n_1,n_2,\\ldots,n_k,p)\\) is
                returned
        random : numpy.random.RandomState
            the random number generator to use (default: `numpy.random`)
        
        Returns
        -------
        destinations : number or numpy.ndarray
            either single value or array of values. See documentation on
            `shape` above for the type of the returned value
        """
        none_shape = (type(shape) is type(None))
        if none_shape:
            return source + self.distribution.draw(random=random)
        elif type(shape) in int_types:
            shape = (shape,)
        if self.numparams == 1:
            return source + self.distribution.draw(shape=shape, random=random)
        else:
            return source[((np.newaxis,) * len(shape)) + (slice(None),)] +\
                self.distribution.draw(shape=shape, random=random)
    
    def log_value(self, source, destination):
        """
        Computes the log-PDF of jumping from `source` to `destination`.
        
        Parameters
        ----------
        source : number or numpy.ndarray
            - if this distribution is univariate, `source` must be a number
            - if this distribution describes \\(p\\) parameters, `source` must
            be a 1D `numpy.ndarray` of length \\(p\\)
        destination : number or numpy.ndarray
            - if this distribution is univariate, `destination` must be a
            number
            - if this distribution describes \\(p\\) parameters, `destination`
            must be a 1D `numpy.ndarray` of length \\(p\\)
        
        Returns
        -------
        log_pdf : float
            if the underlying distribution of this
            `LocaleIndependentJumpingDistribution` is \\(g(\\boldsymbol{z})\\),
            `source` is \\(\\boldsymbol{x}\\) and `destination` is
            \\(\\boldsymbol{y}\\), then `log_pdf` is given by
            \\(\\ln{g(\\boldsymbol{y}-\\boldsymbol{x})}\\)
        """
        return self.distribution.log_value(destination - source)
    
    def log_value_difference(self, source, destination):
        """
        Computes the difference in the log-PDF of jumping from `source` to
        `destination` and of jumping from `destination` to `source`.
        
        Parameters
        ----------
        source : number or numpy.ndarray
            - if this distribution is univariate, `source` must be a number
            - if this distribution describes \\(p\\) parameters, `source` must
            be a 1D `numpy.ndarray` of length \\(p\\)
        destination : number or numpy.ndarray
            - if this distribution is univariate, `destination` must be a
            number
            - if this distribution describes \\(p\\) parameters, `destination`
            must be a 1D `numpy.ndarray` of length \\(p\\)
        
        Returns
        -------
        log_pdf : float
            if the underlying distribution of this
            `LocaleIndependentJumpingDistribution` is \\(g(\\boldsymbol{z})\\),
            `source` is \\(\\boldsymbol{x}\\) and `destination` is
            \\(\\boldsymbol{y}\\), then `log_pdf` is given by
            \\(\\ln{g(\\boldsymbol{y}-\\boldsymbol{x})} -\
            \\ln{g(\\boldsymbol{x}-\\boldsymbol{y})}\\)
        """
        displacement = destination - source
        return self.distribution.log_value(displacement) -\
            self.distribution.log_value(-displacement)
    
    @property
    def numparams(self):
        """
        The integer number of parameters described by this distribution.
        """
        return self.distribution.numparams
    
    def __eq__(self, other):
        """
        Tests for equality between this jumping distribution and other.
        
        Parameters
        ----------
        other : object
            object to check for equality
        
        Returns
        -------
        result : bool
            True if and only if `other` is another
            `LocaleIndependentJumpingDistribution` with the same
            `LocaleIndependentJumpingDistribution.distribution`
        """
        if isinstance(other, LocaleIndependentJumpingDistribution):
            return (self.distribution == other.distribution)
        else:
            return False
    
    @property
    def is_discrete(self):
        """
        Boolean describing whether this `LocaleIndependentJumpingDistribution`
        describes discrete (True) or continuous (False) variable(s).
        """
        return self.distribution.is_discrete
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with information about this jumping
        distribution.
        
        Parameters
        ----------
        group : h5py.Group
            hdf5 file group to fill with information about this jumping
            distribution
        """
        group.attrs['class'] = 'LocaleIndependentJumpingDistribution'
        self.distribution.fill_hdf5_group(group.create_group('distribution'))
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a `LocaleIndependentJumpingDistribution` from the given hdf5 file
        group.
        
        Parameters
        ----------
        group : h5py.Group
            the same hdf5 file group on which
            `LocaleIndependentJumpingDistribution.fill_hdf5_group` was called
        
        Returns
        -------
        loaded: `LocaleIndependentJumpingDistribution`
            `LocaleIndependentJumpingDistribution` loaded from information in
            the given group
        """
        try:
            assert\
                group.attrs['class'] == 'LocaleIndependentJumpingDistribution'
        except:
            raise ValueError("The given group does not seem to contain a " +\
                "LocaleIndependentJumpingDistribution.")
        distribution = load_distribution_from_hdf5_group(group['distribution'])
        return LocaleIndependentJumpingDistribution(distribution)

