"""
Module containing class representing a univariate jumping distribution whose
translation is Gaussian distributed with the extra condition that the
destination cannot fall outside a given range.

**File**: $DISTPY/distpy/jumping/TruncatedGaussianJumpingDistribution.py  
**Author**: Keith Tauscher  
**Date**: 11 Jul 2021
"""
import numpy as np
import numpy.linalg as la
from scipy.special import erf, erfinv
from ..util import numerical_types, sequence_types
from .JumpingDistribution import JumpingDistribution

class TruncatedGaussianJumpingDistribution(JumpingDistribution):
    """
    Class representing a univariate jumping distribution whose translation is
    Gaussian distributed with the extra condition that the destination cannot
    fall outside a given range.
    """
    def __init__(self, variance, low=None, high=None):
        """
        Initializes a `TruncatedGaussianJumpingDistribution` with the given
        variance and endpoints.
        
        Parameters
        ----------
        variance : float
            a single number representing the variance of the non-truncated
            Gaussian
        low : float or None
            - if None, the variate can be an arbitrarily large negative number
            - if float, gives the lowest possible value of the variate
        high : float or None
            - if None, the variate can be an arbitrarily large positive number
            - if float, gives the highest possible value of the variate and
            must be larger than low
        """
        self.variance = variance
        self.low = low
        self.high = high
    
    @property
    def low(self):
        """
        The low endpoint of this truncated Gaussian. Can be -inf
        """
        if not hasattr(self, '_low'):
            raise AttributeError("low referenced before it was set.")
        return self._low
    
    @low.setter
    def low(self, value):
        """
        Setter for `TruncatedGaussianJumpingDistribution.low`.
        
        Parameters
        ----------
        value : float or None
            - if None, the variate can be an arbitrarily large negative number
            - if float, gives the lowest possible value of the variate
        """
        if type(value) is type(None):
            self._low = -np.inf
        elif type(value) in numerical_types:
            self._low = value
        else:
            raise TypeError("low was neither None nor a number.")
    
    @property
    def high(self):
        """
        The low endpoint of this truncated Gaussian. Can be +inf
        """
        if not hasattr(self, '_high'):
            raise AttributeError("high referenced before it was set.")
        return self._high
    
    @high.setter
    def high(self, value):
        """
        Setter for `TruncatedGaussianJumpingDistribution.high`.
        
        Parameters
        ----------
        value : float or None
            - if None, the variate can be an arbitrarily large positive number
            - if float, gives the highest possible value of the variate
        """
        if type(value) is type(None):
            self._high = np.inf
        elif type(value) in numerical_types:
            if value > self.low:
                self._high = value
            else:
                raise ValueError("high was not larger than low.")
        else:
            raise TypeError("high was neither None nor a number.")
    
    @property
    def variance(self):
        """
        The variance, \\(\\sigma^2\\), of the non-truncated Gaussian.
        """
        if not hasattr(self, '_variance'):
            raise AttributeError("variance referenced before it was set.")
        return self._variance
    
    @variance.setter
    def variance(self, value):
        """
        Setter for `TruncatedGaussianJumpingDistribution.variance`.
        
        Parameters
        ----------
        value : float
            a positive number
        """
        if type(value) in numerical_types:
            self._variance = value
        else:
            raise TypeError("variance was not a number.")
    
    @property
    def root_twice_variance(self):
        """
        The square root of twice the variance, \\(\\sqrt{2}\\sigma\\).
        """
        if not hasattr(self, '_root_twice_variance'):
            self._root_twice_variance = np.sqrt(2 * self.variance)
        return self._root_twice_variance
    
    def left_erf(self, source):
        """
        Computes the relevant error function evaluated at `source`
        
        Parameters
        ----------
        source : float
            the mean of the truncated Gaussian
        
        Returns
        -------
        left_erf_value : float
            if `TruncatedGaussianJumpingDistribution.low` is \\(l\\), `source`
            is \\(\\mu\\), and variance \\(\\sigma^2\\), then `left_erf_value`
            is \\(\\text{erf}\\left(\\frac{l-\\mu}{\\sqrt{2}\\sigma}\\right)\\)
        """
        if self.low == -np.inf:
            return (-1.)
        else:
            return erf((self.low - source) / self.root_twice_variance)
    
    def right_erf(self, source):
        """
        Computes the relevant error function evaluated at `source`
        
        Parameters
        ----------
        source : float
            the mean of the truncated Gaussian
        
        Returns
        -------
        right_erf_value : float
            if `TruncatedGaussianJumpingDistribution.high` is \\(h\\), `source`
            is \\(\\mu\\), and variance \\(\\sigma^2\\), then `left_erf_value`
            is \\(\\text{erf}\\left(\\frac{h-\\mu}{\\sqrt{2}\\sigma}\\right)\\)
        """
        if self.high == np.inf:
            return 1.
        else:
            return erf((self.high - source) / self.root_twice_variance)
    
    def erf_difference(self, source):
        """
        Computes the difference of the two error function values.
        right_erf(source)-left_erf(source)
        
        Parameters
        ----------
        source : float
            the mean of the truncated Gaussian
        
        Returns
        -------
        erf_difference_value : float
            if `TruncatedGaussianJumpingDistribution.high` is \\(h\\),
            `TruncatedGaussianJumpingDistribution.low` is \\(l\\), `source` is
            \\(\\mu\\), and variance \\(\\sigma^2\\), then
            `erf_value_difference` is
            \\(\\text{erf}\\left(\\frac{h-\\mu}{\\sqrt{2}\\sigma}\\right)-\
            \\text{erf}\\left(\\frac{l-\\mu}{\\sqrt{2}\\sigma}\\right)\\)
        """
        return (self.right_erf(source) - self.left_erf(source))
    
    @property
    def constant_in_log_value(self):
        """
        A constant in the log value which is independent of both the source and
        the destination.
        """
        if not hasattr(self, '_constant_in_log_value'):
            self._constant_in_log_value =\
                (np.log(2. / (np.pi * self.variance)) / 2.)
        return self._constant_in_log_value
    
    def draw(self, source, shape=None, random=np.random):
        """
        Draws a destination point from this jumping distribution given a source
        point.
        
        Parameters
        ----------
        source : float
            source point
        shape : None or int or tuple
            - if None, a single destination is returned as a single number
            - if int \\(n\\), \\(n\\) destinations are returned as a 1D
            `numpy.ndarray` of length \\(n\\)
            - if tuple of ints \\((n_1,n_2,\\ldots,n_k)\\),
            \\(\\prod_{m=1}^kn_m\\) destinations are returned as a
            `numpy.ndarray` of shape \\((n_1,n_2,\\ldots,n_k)\\)
        random : numpy.random.RandomState
            the random number generator to use (default: `numpy.random`)
        
        Returns
        -------
        drawn : number or numpy.ndarray
            either single value or array of values. See documentation on
            `shape` above for the type of the returned value
        """
        uniforms = random.uniform(size=shape)
        erfinv_argument = ((uniforms * self.right_erf(source)) +\
            ((1 - uniforms) * self.left_erf(source)))
        return (source + (self.root_twice_variance * erfinv(erfinv_argument)))
    
    def log_value(self, source, destination):
        """
        Computes the log-PDF of jumping from `source` to `destination`.
        
        Parameters
        ----------
        source : float
            source point
        destination : float
            destination point
        
        Returns
        -------
        log_pdf : float
            if the distribution is \\(f(x,y)=\\text{Pr}[y|x]\\), `source` is
            \\(x\\) and `destination` is \\(y\\), then `log_pdf` is given by
            \\(\\ln{f(x,y)}\\)
        """
        difference = (destination - source)
        return (self.constant_in_log_value +\
            (((difference / self.standard_deviation) ** 2) / (-2.))) -\
            np.log(self.erf_difference(source))
    
    def log_value_difference(self, source, destination):
        """
        Computes the difference in the log-PDF of jumping from `source` to
        `destination` and of jumping from `destination` to `source`. While this
        method has a default version, overriding it may provide an efficiency
        benefit.
        
        Parameters
        ----------
        source : float
            source point
        destination : float
            destination point
        
        Returns
        -------
        log_pdf_difference : float
            if the distribution is \\(f(x,y)=\\text{Pr}[y|x]\\), `source` is
            \\(x\\) and `destination` is \\(y\\), then `log_pdf_difference` is
            given by \\(\\ln{f(x,y)}-\\ln{f(y,x)}\\)
        """
        return np.log(\
            self.erf_difference(destination) / self.erf_difference(source))
    
    @property
    def numparams(self):
        """
        The integer number of parameters described by this distribution. Since
        the truncated Gaussian is only easily analytically sampled in the case
        of 1 parameter, `TruncatedGaussianJumpingDistribution` only allows one
        parameter.
        """
        return 1
    
    @property
    def standard_deviation(self):
        """
        The square root of the variance.
        """
        if not hasattr(self, '_standard_deviation'):
            self._standard_deviation = np.sqrt(self.variance)
        return self._standard_deviation
    
    def __eq__(self, other):
        """
        Tests for equality between this `TruncatedGaussianJumpingDistribution`
        and `other`.
        
        Parameters
        ----------
        other : object
            object with which to check for equality
        
        Returns
        -------
        result : bool
            True if and only if object is another
            `TruncatedGaussianJumpingDistribution` with the same
            `TruncatedGaussianJumpingDistribution.variance`,
            `TruncatedGaussianJumpingDistribution.low`, and
            `TruncatedGaussianJumpingDistribution.high`
        """
        if isinstance(other, TruncatedGaussianJumpingDistribution):
            if self.numparams == other.numparams:
                variances_close = np.allclose(self.variance, other.variance,\
                    rtol=1e-12, atol=1e-12)
                lows_close = np.isclose(self.low, other.low, atol=1e-6)
                highs_close = np.isclose(self.high, other.high, atol=1e-6)
                return (variances_close and lows_close and highs_close)
            else:
                return False
        else:
            return False
    
    @property
    def is_discrete(self):
        """
        Boolean describing whether this `TruncatedGaussianJumpingDistribution`
        describes discrete (True) or continuous (False) variable(s). Since
        Gaussian distributions are continuous, this is always False.
        """
        return False
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data from this distribution.
        
        Parameters
        ----------
        group : h5py.Group
            hdf5 file group to fill with information about this distribution
        """
        group.attrs['class'] = 'TruncatedGaussianJumpingDistribution'
        group.attrs['variance'] = self.variance
        if self.low != -np.inf:
            group.attrs['low'] = self.low
        if self.high != np.inf:
            group.attrs['high'] = self.high
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a `TruncatedGaussianJumpingDistribution` from the given hdf5 file
        group.
        
        Parameters
        ----------
        group : h5py.Group
            the same hdf5 file group which
            `TruncatedGaussianJumpingDistribution.fill_hdf5_group` was called
            on
        
        Returns
        -------
        loaded : `TruncatedGaussianJumpingDistribution`
            distribution loaded from information in the given group
        """
        try:
            assert\
                group.attrs['class'] == 'TruncatedGaussianJumpingDistribution'
        except:
            raise ValueError("The given group does not seem to contain a " +\
                "TruncatedGaussianJumpingDistribution.")
        variance = group.attrs['variance']
        if 'low' in group.attrs:
            low = group.attrs['low']
        else:
            low = None
        if 'high' in group.attrs:
            high = group.attrs['high']
        else:
            high = None
        return\
            TruncatedGaussianJumpingDistribution(variance, low=low, high=high)

