"""
Module containing class representing a jumping distribution which describes a
distribution whose destination does not depend on the starting location. If the
PDF of the destination distribution is \\(g(\\boldsymbol{y})\\), this
distribution's PDF is given by
$$f(\\boldsymbol{x},\\boldsymbol{y})=g(\\boldsymbol{y})$$

**File**: $DISTPY/distpy/jumping/SourceIndependentJumpingDistribution.py  
**Author**: Keith Tauscher  
**Date**: 11 Jul 2021
"""
import numpy as np
from ..distribution import Distribution, load_distribution_from_hdf5_group
from .JumpingDistribution import JumpingDistribution

class SourceIndependentJumpingDistribution(JumpingDistribution):
    """
    Class representing a jumping distribution which describes a distribution
    whose destination does not depend on the starting location. If the PDF of
    the destination distribution is \\(g(\\boldsymbol{y})\\), this
    distribution's PDF is given by
    $$f(\\boldsymbol{x},\\boldsymbol{y})=g(\\boldsymbol{y})$$
    """
    def __init__(self, distribution):
        """
        Initializes this SourceIndependentJumpingDistribution with the given
        core distribution.
        
        distribution: a Distribution object describing the probability of
                      jumping to any destination, regardless of source
        """
        self.distribution = distribution
    
    @property
    def distribution(self):
        """
        The distribution of the destination, regardless of the source.
        """
        if not hasattr(self, '_distribution'):
            raise AttributeError("distribution referenced before it was set.")
        return self._distribution
    
    @distribution.setter
    def distribution(self, value):
        """
        Setter for `SourceIndependentJumpingDistribution.distribution`.
        
        Parameters
        ----------
        value : `distpy.distribution.Distribution.Distribution`
            a distribution describing the probability of jumping to any
            destination regardless of source
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
            - if this distribution is univariate, source should be a single
            number
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
        return self.distribution.draw(shape=shape, random=random)
    
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
            if the core distribution is \\(g(\\boldsymbol{y})=\
            \\text{Pr}[\\boldsymbol{y}|\\boldsymbol{x}]\\), `source` is
            \\(\\boldsymbol{x}\\) and `destination` is \\(\\boldsymbol{y}\\),
            then `log_pdf` is given by \\(\\ln{g(\\boldsymbol{y})}\\)
        """
        return self.distribution.log_value(destination)
    
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
        log_pdf_difference : float
            if the core distribution is \\(g(\\boldsymbol{y})=\
            \\text{Pr}[\\boldsymbol{y}|\\boldsymbol{x}]\\), `source` is
            \\(\\boldsymbol{x}\\) and `destination` is \\(\\boldsymbol{y}\\),
            then `log_pdf_difference` is given by
            \\(\\ln{g(\\boldsymbol{y})}-\\ln{g(\\boldsymbol{x})}\\)
        """
        return self.distribution.log_value(destination) -\
            self.distribution.log_value(source)
    
    @property
    def numparams(self):
        """
        The integer number of parameters described by this distribution.
        """
        return self.distribution.numparams
    
    def __eq__(self, other):
        """
        Tests for equality between this `SourceIndependentJumpingDistribution`
        and `other`.
        
        Parameters
        ----------
        other : object
            object with which to check for equality
        
        Returns
        -------
        result : bool
            True if and only if object is another
            `SourceIndependentJumpingDistribution` with the same
            `SourceIndependentJumpingDistribution.distribution`
        """
        if isinstance(other, SourceIndependentJumpingDistribution):
            return (self.distribution == other.distribution)
        else:
            return False
    
    @property
    def is_discrete(self):
        """
        Boolean describing whether this `SourceIndependentJumpingDistribution`
        describes discrete (True) or continuous (False) variable(s). This has
        the same value as the
        `distpy.distribution.Distribution.Distribution.is_discrete` property of
        `SourceIndependentJumpingDistribution.distribution`
        """
        return self.distribution.is_discrete
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with information about this jumping
        distribution.
        
        Parameters
        ----------
        group : h5py.Group
            hdf5 file group to fill with information about this distribution
        """
        group.attrs['class'] = 'SourceIndependentJumpingDistribution'
        self.distribution.fill_hdf5_group(group.create_group('distribution'))
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a `SourceIndependentJumpingDistribution` from the given hdf5 file
        group.
        
        Parameters
        ----------
        group : h5py.Group
            the same hdf5 file group which
            `SourceIndependentJumpingDistribution.fill_hdf5_group` was called
            on
        
        Returns
        -------
        loaded : `SourceIndependentJumpingDistribution`
            distribution loaded from the information in the given group
        """
        try:
            assert\
                group.attrs['class'] == 'SourceIndependentJumpingDistribution'
        except:
            raise ValueError("The given group does not seem to contain a " +\
                "SourceIndependentJumpingDistribution.")
        distribution = load_distribution_from_hdf5_group(group['distribution'])
        return SourceIndependentJumpingDistribution(distribution)

