"""
Module containing class representing a jumping distribution based on the
binomial distribution. Its PMF is given by $$(y-\\text{min})|(x-\\text{min})\
\\sim \\begin{cases} \\text{Binomial}\\left(\\text{max}-\\text{min},\
\\frac{x-\\text{min}}{\\text{max}-\\text{min}}\\right) &\
\\text{min}<x<\\text{max} \\\\ \\text{Binomial}\\left(\\text{max}-\\text{min},\
\\frac{1}{2(\\text{max}-\\text{min})}\\right) & x=\\text{min} \\\\\
\\text{Binomial}\\left(\\text{max}-\\text{min},\
1-\\frac{1}{2(\\text{max}-\\text{min})}\\right)& x=\\text{max} \\end{cases}$$

**File**: $DISTPY/distpy/jumping/BinomialJumpingDistribution.py  
**Author**: Keith Tauscher  
**Date**: 3 Jul 2021
"""
import numpy as np
from scipy.special  import gammaln as log_gamma
from ..util import int_types
from .JumpingDistribution import JumpingDistribution

class BinomialJumpingDistribution(JumpingDistribution):
    """
    Class representing a jumping distribution based on the binomial
    distribution. Its PMF is given by $$(y-\\text{min})|(x-\\text{min}) \\sim\
    \\begin{cases} \\text{Binomial}\\left(\\text{max}-\\text{min},\
    \\frac{x-\\text{min}}{\\text{max}-\\text{min}}\\right) &\
    \\text{min}<x<\\text{max} \\\\ \\text{Binomial}\
    \\left(\\text{max}-\\text{min},\
    \\frac{1}{2(\\text{max}-\\text{min})}\\right) & x=\\text{min} \\\\\
    \\text{Binomial}\\left(\\text{max}-\\text{min},\
    1-\\frac{1}{2(\\text{max}-\\text{min})}\\right)& x=\\text{max}\
    \\end{cases}$$
    """
    def __init__(self, minimum, maximum):
        """
        Initializes a `BinomialJumpingDistribution` with the given extrema.
        
        Parameters
        ----------
        minimum : int
            minimum allowable value of integer parameter
        maximum : int
            maximum allowable value of integer parameter
        """
        self.minimum = minimum
        self.maximum = maximum
    
    @property
    def minimum(self):
        """
        The minimum allowable integer value of the parameter.
        """
        if not hasattr(self, '_minimum'):
            raise AttributeError("minimum was referenced before it was set.")
        return self._minimum
    
    @minimum.setter
    def minimum(self, value):
        """
        Setter for `BinomialJumpingDistribution.minimum`.
        
        Parameters
        ----------
        value : int
            minimum allowable value of integer parameter
        """
        if type(value) in int_types:
            self._minimum = value
        else:
            raise TypeError("minimum was set to a non-int.")
    
    @property
    def maximum(self):
        """
        The maximum allowable integer value of the parameter.
        """
        if not hasattr(self, '_maximum'):
            raise AttributeError("maximum was referenced before it was set.")
        return self._maximum
    
    @maximum.setter
    def maximum(self, value):
        """
        Setter for `BinomialJumpingDistribution.maximum`.
        
        Parameters
        ----------
        value : int
            maximum allowable value of integer parameter
        """
        if type(value) in int_types:
            if value > self.minimum:
                self._maximum = value
            else:
                raise ValueError("maximum was set to an int which was less " +\
                    "than or equal to minimum.")
        else:
            raise TypeError("maximum was set to a non-int.")
    
    @property
    def span(self):
        """
        The difference between the minimum and maximum allowed value of the
        parameter, \\(\\text{max}-\\text{min}\\).
        """
        if not hasattr(self, '_span'):
            self._span = (self.maximum - self.minimum)
        return self._span
    
    @property
    def reciprocal_span(self):
        """
        The reciprocal of the span property,
        \\(\\frac{1}{\\text{max}-\\text{min}}\\).
        """
        if not hasattr(self, '_reciprocal_span'):
            self._reciprocal_span = (1. / self.span)
        return self._reciprocal_span
    
    @property
    def half_reciprocal_span(self):
        """
        Half of the reciprocal of the span property,
        \\(\\frac{1}{2(\\text{max}-\\text{min})}\\).
        """
        if not hasattr(self, '_half_reciprocal_span'):
            self._half_reciprocal_span = self.reciprocal_span / 2
        return self._half_reciprocal_span
    
    def p_from_shifted_source(self, shifted_source):
        """
        Finds the value of the \\(p\\) parameter of the binomial distribution
        whose mean is the given source (which is assumed to already have
        `BinomialJumpingDistribution.minimum` subtracted)
        
        Parameters
        ----------
        shifted_source : int
            integer between 0 and `BinomialJumpingDistribution.span`
            (inclusive)
        
        Returns
        -------
        probability_of_success : float
            p satisfying \\(0 < p < 1\\) where \\(pN\\) is near (and usually
            equal to) shifted_source (p cannot be 0 or 1 because that would
            imply it could never jump away from the minimum or maximum, and
            would therefore violate the ergodicity requirement of Markov
            chains)
        """
        if shifted_source == 0:
            return self.half_reciprocal_span
        elif shifted_source == self.span:
            return (1 - self.half_reciprocal_span)
        else:
            return (shifted_source * self.reciprocal_span)
    
    def draw(self, source, shape=None, random=np.random):
        """
        Draws a destination point from this jumping distribution given a source
        point.
        
        Parameters
        ----------
        source : int
            integer source point
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
        return self.minimum + random.binomial(self.span,\
            self.p_from_shifted_source(source - self.minimum), size=shape)
    
    @property
    def log_value_constant(self):
        """
        A constant in the log value of this distribution which is independent
        of the source and destination integers.
        """
        if not hasattr(self, '_log_value_constant'):
            self._log_value_constant = log_gamma(self.span + 1)
        return self._log_value_constant
    
    def log_value(self, source, destination):
        """
        Computes the log-PDF of jumping from `source` to `destination`.
        
        Parameters
        ----------
        source : int
            source integer
        destination : int
            destination integer
        
        Returns
        -------
        log_pdf : float
            if the distribution is \\(f(x,y)=\\text{Pr}[y|x]\\), `source` is
            \\(x\\) and `destination` is \\(y\\), then `log_pdf` is given by
            \\(\\ln{f(x,y)}\\)
        """
        shifted_source = source - self.minimum
        shifted_destination = destination - self.minimum
        p_parameter = self.p_from_shifted_source(shifted_source)
        return self.log_value_constant - log_gamma(shifted_destination + 1) -\
            log_gamma(self.span - shifted_destination + 1) +\
            (shifted_destination * np.log(p_parameter)) +\
            ((self.span - shifted_destination) * np.log(1 - p_parameter))
    
    @property
    def numparams(self):
        """
        The integer number of parameters described by this distribution. Since
        this distribution is univariate, this property is always 1.
        """
        return 1
    
    def __eq__(self, other):
        """
        Tests for equality between this jumping distribution and other.
        
        Parameters
        ----------
        other : object
            object with which to check for equality
        
        Returns
        -------
        result : bool
            True if and only if `other` is an `BinomialJumpingDistribution`
            with the same `BinomialJumpingDistribution.minimum` and
            `BinomialJumpingDistribution.maximum`
        """
        if isinstance(other, BinomialJumpingDistribution):
            return (self.minimum == other.minimum) and\
                (self.maximum == other.maximum)
        else:
            return False
    
    @property
    def is_discrete(self):
        """
        Boolean describing whether this `BinomialJumpingDistribution` describes
        discrete (True) or continuous (False) variable(s). Since this is a
        discrete distribution, it is always True.
        """
        return True
    
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
        group.attrs['class'] = 'BinomialJumpingDistribution'
        group.attrs['minimum'] = self.minimum
        group.attrs['maximum'] = self.maximum
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a `BinomialJumpingDistribution` from the given hdf5 file group.
        
        Parameters
        ----------
        group : h5py.Group
            the same hdf5 file group which
            `BinomialJumpingDistribution.fill_hdf5_group` was called on
        
        Returns
        -------
        loaded : `BinomialJumpingDistribution`
            `BinomialJumpingDistribution` object loaded from the given group
        """
        try:
            assert group.attrs['class'] == 'BinomialJumpingDistribution'
        except:
            raise ValueError("The given group does not seem to contain a " +\
                "BinomialJumpingDistribution.")
        minimum = group.attrs['minimum']
        maximum = group.attrs['maximum']
        return BinomialJumpingDistribution(minimum, maximum)

