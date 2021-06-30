"""
Module containing class representing a binomial distribution. Its PMF is
represented by:
$$f(x) = \\begin{pmatrix} n \\\\ x \\end{pmatrix} p^x(1-p)^{n-x}$$

**File**: $DISTPY/distpy/distribution/BinomialDistribution.py  
**Author**: Keith Tauscher  
**Date**: 31 May 2021
"""
import numpy as np
import numpy.random as rand
from scipy.special import gammaln as log_gamma
from ..util import int_types, numerical_types
from .Distribution import Distribution

class BinomialDistribution(Distribution):
    """
    Class representing a binomial distribution. Its PMF is represented by:
    $$f(x) = \\begin{pmatrix} n \\\\ x \\end{pmatrix} p^x(1-p)^{n-x}$$
    """
    def __init__(self, probability_of_success, number_of_trials,\
        metadata=None):
        """
        Initializes a new `BinomialDistribution` with the given parameter
        values.
        
        Parameters
        ----------
        probability_of_success : float
            real number, \\(p\\), in (0, 1)
        number_of_trials : int
            positive integer, \\(n\\)
        metadata : number or str or dict or `distpy.util.Savable.Savable`
            data to store alongside this distribution.
        """
        self.probability_of_success = probability_of_success
        self.number_of_trials = number_of_trials
        self.metadata = metadata
    
    @property
    def number_of_trials(self):
        """
        The integer number of trials, \\(n\\), from which to draw numbers of
        successes.
        """
        if not hasattr(self, '_number_of_trials'):
            raise AttributeError("number_of_trials was referenced before " +\
                "it was set.")
        return self._number_of_trials
    
    @number_of_trials.setter
    def number_of_trials(self, value):
        """
        Setter for `BinomialDistribution.number_of_trials`
        
        Parameters
        ----------
        value : int
            positive integer
        """
        if type(value) in int_types:
            if value > 0:
                self._number_of_trials = value
            else:
                raise ValueError("number_of_trials given to " +\
                    "BinomialDistribution was not positive.")
        else:
            raise ValueError("number_of_trials given to " +\
                "BinomialDistribution was not a number.")
    
    @property
    def probability_of_success(self):
        """
        The probability of a success on a given trial, \\(p\\).
        """
        if not hasattr(self, '_probability_of_success'):
            raise AttributeError("probability_of_success was referenced " +\
                "before it was set.")
        return self._probability_of_success
    
    @probability_of_success.setter
    def probability_of_success(self, value):
        """
        Setter for `BinomialDistribution.probability_of_success`.
        
        Parameters
        ----------
        value : float
            real number between 0 and 1 (exclusive)
        """
        if type(value) in numerical_types:
            if (value > 0.) and (value < 1.):
                self._probability_of_success = value
            else:
                raise ValueError("probability_of_success given to " +\
                    "BinomialDistribution was not between 0 and 1.")
        else:
            raise ValueError("probability_of_success given to " +\
                "BinomialDistribution was not a number.")
    
    @property
    def numparams(self):
        """
        The number of parameters of this `BinomialDistribution`, 1.
        """
        return 1
    
    @property
    def mean(self):
        """
        The mean of this `BinomialDistribution`, \\(np\\).
        """
        if not hasattr(self, '_mean'):
            self._mean = self.number_of_trials * self.probability_of_success
        return self._mean
    
    @property
    def variance(self):
        """
        The variance of this `BinomialDistribution`, \\(np(1-p)\\).
        """
        if not hasattr(self, '_variance'):
            self._variance = self.number_of_trials *\
                self.probability_of_success * (1 - self.probability_of_success)
        return self._variance
    
    def draw(self, shape=None, random=rand):
        """
        Draws point(s) from this `BinomialDistribution`.
        
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
        return random.binomial(self.number_of_trials,\
            self.probability_of_success, size=shape)
    
    def log_value(self, point):
        """
        Computes the logarithm of the value of this `BinomialDistribution` at
        the given point.
        
        Parameters
        ----------
        point : int
            scalar at which to evaluate PDF
        
        Returns
        -------
        value : float
            natural logarithm of the value of this distribution at `point`. If
            \\(f\\) is this distribution's PDF and \\(x\\) is `point`, then
            `value` is \\(\\ln{\\big(f(x)\\big)}\\)
        """
        if type(point) in int_types:
            if (point >= 0) and (point <= self.number_of_trials):
                n_minus_k = self.number_of_trials - point
                return log_gamma(self.number_of_trials + 1) -\
                    log_gamma(point + 1) - log_gamma(n_minus_k + 1) +\
                    (point * np.log(self.probability_of_success)) +\
                    (n_minus_k * np.log(1 - self.probability_of_success))
            else:
                return -np.inf
        else:
            raise TypeError("point given to BinomialDistribution was not " +\
                "an integer.")

    def to_string(self):
        """
        Finds and returns a string version of this `BernoulliDistribution` of
        the form `"Bernoulli(p,n)"`.
        """
        return "Binomial({0:.2g},{1:d})".format(self.probability_of_success,\
            self.number_of_trials)
    
    def __eq__(self, other):
        """
        Checks for equality of this `BinomialDistribution` with `other`.
        
        Parameters
        ----------
        other : object
            object to check for equality
        
        Returns
        -------
        result : bool
            True if and only if `other` is a `BinomialDistribution` with the
            same `BinomialDistribution.probability_of_success` and
            `BinomialDistribution.number_of_trials`
        """
        if isinstance(other, BinomialDistribution):
            p_close = np.isclose(self.probability_of_success,\
                other.probability_of_success, rtol=0, atol=1e-6)
            n_equal = (self.number_of_trials == other.number_of_trials)
            metadata_equal = self.metadata_equal(other)
            return all([p_close, n_equal, metadata_equal])
        else:
            return False
    
    @property
    def can_give_confidence_intervals(self):
        """
        Discrete distributions do not support confidence intervals.
        """
        return False
    
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
        return self.number_of_trials
    
    @property
    def is_discrete(self):
        """
        Boolean describing whether this distribution is discrete (True) or
        continuous (False).
        """
        return True
    
    def fill_hdf5_group(self, group, save_metadata=True):
        """
        Fills the given hdf5 file group with data about this
        `BinomialDistribution` so that it can be loaded later.
        
        Parameters
        ----------
        group : h5py.Group
            hdf5 file group to fill
        save_metadata : bool
            - if True, attempts to save metadata alongside distribution and
            throws error if it fails
            - if False, metadata is ignored in saving process
        """
        group.attrs['class'] = 'BinomialDistribution'
        group.attrs['number_of_trials'] = self.number_of_trials
        group.attrs['probability_of_success'] = self.probability_of_success
        if save_metadata:
            self.save_metadata(group)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a `BinomialDistribution` from the given hdf5 file group.
        
        Parameters
        ----------
        group : h5py.Group
            the same hdf5 file group which fill_hdf5_group was called on when
            this Distribution was saved
        
        Returns
        -------
        distribution : `BinomialDistribution`
            distribution created from the information in the given group
        """
        try:
            assert group.attrs['class'] == 'BinomialDistribution'
        except:
            raise TypeError("The given hdf5 file doesn't seem to contain a " +\
                "BinomialDistribution.")
        metadata = Distribution.load_metadata(group)
        probability_of_success = group.attrs['probability_of_success']
        number_of_trials = group.attrs['number_of_trials']
        return BinomialDistribution(probability_of_success, number_of_trials,\
            metadata=metadata)
    
    @property
    def gradient_computable(self):
        """
        Boolean describing whether the gradient of the given distribution has
        been implemented. If True,
        `BinomialDistribution.gradient_of_log_value` method can be called
        safely.
        """
        return False 
    
    @property
    def hessian_computable(self):
        """
        Boolean describing whether the hessian of the given distribution has
        been implemented. If True,
        `BinomialDistribution.hessian_of_log_value` method can be called
        safely.
        """
        return False
    
    def copy(self):
        """
        Copies this distribution.
        
        Returns
        -------
        copied : `BinomialDistribution`
            a deep copy of this distribution, ignoring metadata.
        """
        return BinomialDistribution(self.probability_of_success,\
            self.number_of_trials)

