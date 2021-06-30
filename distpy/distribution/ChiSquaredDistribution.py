"""
Module containing class representing a chi-squared distribution. Its PDF is
represented by: $$f(x) = \\frac{x^{k/2}\\ e^{-x/2}}{2^{k/2}\\ \\Gamma(k/2)},$$
where \\(\\Gamma(x)\\) is the Gamma function.

**File**: $DISTPY/distpy/distribution/ChiSquaredDistribution.py  
**Author**: Keith Tauscher  
**Date**: 31 May 2021
"""
from __future__ import division
import numpy as np
import numpy.random as rand
from scipy.special import gammaln, gammaincinv
from ..util import bool_types, int_types
from .Distribution import Distribution

class ChiSquaredDistribution(Distribution):
    """
    Class representing a chi-squared distribution. Its PDF is represented by:
    $$f(x) = \\frac{x^{k/2}\\ e^{-x/2}}{2^{k/2}\\ \\Gamma(k/2)},$$ where
    \\(\\Gamma(x)\\) is the Gamma function.
    """
    def __init__(self, degrees_of_freedom, reduced=False, metadata=None):
        """
        Initializes a new `ChiSquaredDistribution` with the given parameter
        values.
        
        Parameters
        ----------
        degrees_of_freedom : int
            integer number of degrees of freedom, \\(k\\)
        reduced : bool
            - if True, the distribution is scaled so that the mean is 1
            - if False, the mean of the distribution is \\(k\\)
        metadata : number or str or dict or `distpy.util.Savable.Savable`
            data to store alongside this distribution.
        """
        self.degrees_of_freedom = degrees_of_freedom
        self.reduced = reduced
        self.metadata = metadata
    
    @property
    def degrees_of_freedom(self):
        """
        The number of degrees of freedom, \\(k\\), of the distribution.
        """
        if not hasattr(self, '_degrees_of_freedom'):
            raise AttributeError("degrees_of_freedom was referenced before " +\
                "it was set.")
        return self._degrees_of_freedom
    
    @degrees_of_freedom.setter
    def degrees_of_freedom(self, value):
        """
        Setter for `ChiSquaredDistribution.degrees_of_freedom`.
        
        Parameters
        ----------
        value : int
            positive integer
        """
        if type(value) in int_types:
            if value > 0:
                self._degrees_of_freedom = value
            else:
                raise ValueError("degrees_of_freedom_given to " +\
                    "ChiSquaredDistribution was not positive.")
        else:
            raise ValueError("degrees_of_freedom given to " +\
                "ChiSquaredDistribution was not an integer.")
    
    @property
    def const_lp_term(self):
        """
        The constant part of the logarithm of the probability density of this
        distribution, given by
        \\(-\\ln{\\Gamma\\left(\\frac{k}{2}\\right)}-\\frac{k}{2}\\ln{2} +\
        \\begin{cases} \\ln{k} & \\text{reduced} \\\\ 0 & \\text{otherwise}\
        \\end{cases}\\).
        """
        if not hasattr(self, '_const_lp_term'):
            self._const_lp_term =\
                ((-1) * gammaln(self.degrees_of_freedom / 2)) -\
                (self.degrees_of_freedom * (np.log(2) / 2))
            if self.reduced:
                self._const_lp_term =\
                    self._const_lp_term + np.log(self.degrees_of_freedom)
        return self._const_lp_term
    
    @property
    def reduced(self):
        """
        A boolean determining whether this distribution represents a reduced
        chi squared statistic or not.
        """
        if not hasattr(self, '_reduced'):
            raise AttributeError("reduced referenced before it was set.")
        return self._reduced
    
    @reduced.setter
    def reduced(self, value):
        """
        Setter for `ChiSquaredDistribution.reduced`.
        
        Parameters
        ----------
        value : bool
            True or False
        """
        if type(value) in bool_types:
            self._reduced = value
        else:
            raise TypeError("reduced was set to a non-bool.")
    
    @property
    def numparams(self):
        """
        The number of parameters of this `ChiSquaredDistribution`, 1.
        """
        return 1
    
    @property
    def mean(self):
        """
        The mean of this `ChiSquaredDistribution`, \\(1\\) if
        `ChiSquaredDistribution.reduced`, otherwise \\(d\\).
        """
        if not hasattr(self, '_mean'):
            if self.reduced:
                self._mean = 1.
            else:
                self._mean = (1. * self.degrees_of_freedom)
        return self._mean
    
    @property
    def variance(self):
        """
        The variance of this `ChiSquaredDistribution`, \\(2/d\\) if
        `ChiSquaredDistribution.reduced`, otherwise \\(2d\\).
        """
        if not hasattr(self, '_variance'):
            if self.reduced:
                self._variance = (2. / self.degrees_of_freedom)
            else:
                self._variance = (2. * self.degrees_of_freedom)
        return self._variance

    def draw(self, shape=None, random=rand):
        """
        Draws point(s) from this `ChiSquaredDistribution`.
        
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
        sample = random.chisquare(self.degrees_of_freedom, size=shape)
        if self.reduced:
            return sample / self.degrees_of_freedom
        else:
            return sample

    def log_value(self, point):
        """
        Computes the logarithm of the value of this `ChiSquaredDistribution` at
        the given point.
        
        Parameters
        ----------
        point : float
            scalar at which to evaluate PDF
        
        Returns
        -------
        value : float
            natural logarithm of the value of this distribution at `point`. If
            \\(f\\) is this distribution's PDF and \\(x\\) is `point`, then
            `value` is \\(\\ln{\\big(f(x)\\big)}\\)
        """
        if self.reduced:
            point = point * self.degrees_of_freedom
        return self.const_lp_term - (point / 2) +\
            (((self.degrees_of_freedom / 2.) - 1) * np.log(point))
    
    def to_string(self):
        """
        Finds and returns a string version of this `ChiSquaredDistribution` of
        the form `"ChiSquared(d)"` or `"ReducedChiSquared(d)"`.
        """
        if self.reduced:
            return "ChiSquared({})".format(self.degrees_of_freedom)
        else:
            return "ReducedChiSquared({})".format(self.degrees_of_freedom)
    
    def __eq__(self, other):
        """
        Checks for equality of this `ChiSquaredDistribution` with `other`.
        
        Parameters
        ----------
        other : object
            object to check for equality
        
        Returns
        -------
        result : bool
            True if and only if `other` is a `ChiSquaredDistribution` with the
            same `ChiSquaredDistribution.degrees_of_freedom` and
            `ChiSquaredDistribution.reduced`
        """
        if isinstance(other, ChiSquaredDistribution):
            dof_equal = (self.degrees_of_freedom == other.degrees_of_freedom)
            reduced_equal = (self.reduced == other.reduced)
            metadata_equal = self.metadata_equal(other)
            return all([dof_equal, reduced_equal, metadata_equal])
        else:
            return False
    
    def inverse_cdf(self, cdf):
        """
        Computes the inverse of the cumulative distribution function (cdf) of
        this `ChiSquaredDistribution`.
        
        Parameters
        ----------
        cdf : float
            probability value between 0 and 1
        
        Returns
        -------
        point : float
            value which yields `cdf` when it the CDF is evaluated at it
        """
        answer = 2 * gammaincinv(self.degrees_of_freedom / 2, cdf)
        if self.reduced:
            answer = answer / self.degrees_of_freedom
        return answer
    
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
        return None
    
    @property
    def is_discrete(self):
        """
        Boolean describing whether this distribution is discrete (True) or
        continuous (False).
        """
        return False
    
    def fill_hdf5_group(self, group, save_metadata=True):
        """
        Fills the given hdf5 file group with data about this
        `ChiSquaredDistribution` so that it can be loaded later.
        
        Parameters
        ----------
        group : h5py.Group
            hdf5 file group to fill
        save_metadata : bool
            - if True, attempts to save metadata alongside distribution and
            throws error if it fails
            - if False, metadata is ignored in saving process
        """
        group.attrs['class'] = 'ChiSquaredDistribution'
        group.attrs['degrees_of_freedom'] = self.degrees_of_freedom
        group.attrs['reduced'] = self.reduced
        if save_metadata:
            self.save_metadata(group)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a `ChiSquaredDistribution` from the given hdf5 file group.
        
        Parameters
        ----------
        group : h5py.Group
            the same hdf5 file group which fill_hdf5_group was called on when
            this Distribution was saved
        
        Returns
        -------
        distribution : `ChiSquaredDistribution`
            distribution created from the information in the given group
        """
        try:
            assert group.attrs['class'] == 'ChiSquaredDistribution'
        except:
            raise TypeError("The given hdf5 file doesn't seem to contain a " +\
                "ChiSquaredDistribution.")
        metadata = Distribution.load_metadata(group)
        degrees_of_freedom = group.attrs['degrees_of_freedom']
        reduced = group.attrs['reduced']
        return ChiSquaredDistribution(degrees_of_freedom, reduced=reduced,\
            metadata=metadata)
    
    @property
    def gradient_computable(self):
        """
        Boolean describing whether the gradient of the given distribution has
        been implemented. If True,
        `ChiSquaredDistribution.gradient_of_log_value` method can be called
        safely.
        """
        return True
    
    def gradient_of_log_value(self, point):
        """
        Computes the gradient (derivative) of the logarithm of the value of
        this `ChiSquaredDistribution` at the given point.
        
        Parameters
        ----------
        point : float
            scalar at which to evaluate the gradient
        
        Returns
        -------
        value : float
            gradient of the natural logarithm of the value of this
            distribution. If \\(f\\) is this distribution's PDF and \\(x\\) is
            `point`, then `value` is
            \\(\\boldsymbol{\\nabla}\\ln{\\big(f(x)\\big)}\\) as a float
        """
        constant = 1.
        if self.reduced:
            constant /= self.degrees_of_freedom
        return ((((self.degrees_of_freedom - 2) / point) - constant) / 2.)
    
    @property
    def hessian_computable(self):
        """
        Boolean describing whether the hessian of the given distribution has
        been implemented. If True,
        `ChiSquaredDistribution.hessian_of_log_value` method can be called
        safely.
        """
        return True
    
    def hessian_of_log_value(self, point):
        """
        Computes the hessian (second derivative) of the logarithm of the value
        of this `ChiSquaredDistribution` at the given point.
        
        Parameters
        ----------
        point : float
            scalar at which to evaluate the gradient
        
        Returns
        -------
        value : float
            hessian of the natural logarithm of the value of this
            distribution. If \\(f\\) is this distribution's PDF and \\(x\\) is
            `point`, then `value` is
            \\(\\boldsymbol{\\nabla}\\boldsymbol{\\nabla}^T\
            \\ln{\\big(f(x)\\big)}\\) as a float
        """
        return ((2 - self.degrees_of_freedom) / (2. * (point ** 2)))
    
    def copy(self):
        """
        Copies this distribution.
        
        Returns
        -------
        copied : `ChiSquaredDistribution`
            a deep copy of this distribution, ignoring metadata.
        """
        return ChiSquaredDistribution(self.degrees_of_freedom,\
            reduced=self.reduced)

