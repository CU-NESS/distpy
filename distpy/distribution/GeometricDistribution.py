"""
Module containing class representing a geometric distribution. Its PMF is
represented by: $$f(x)=\\begin{cases}\
\\frac{1-p}{1-p^{x_{\\text{max}}-x_{\\text{min}}+1}}\\ p^{x-x_{\\text{min}}} &\
x \\in \\{x_{\\text{min}},x_{\\text{min}}+1,\\ldots,x_{\\text{max}}\\} \\\\\
0 & \\text{otherwise} \\end{cases},$$ where \\(x_{\\text{min}}\\) is an integer
and \\(x_{\\text{max}}\\) can be either an integer greater than or equal to
\\(x_{\\text{min}}\\) or \\(\\infty\\).

**File**: $DISTPY/distpy/distribution/GeometricDistribution.py  
**Author**: Keith Tauscher  
**Date**: 31 May 2021
"""
import numpy as np
import numpy.random as rand
from ..util import int_types, numerical_types
from .Distribution import Distribution

class GeometricDistribution(Distribution):
    """
    Class representing a geometric distribution. Its PMF is represented by:
    $$f(x)=\\begin{cases}\
    \\frac{1-p}{1-p^{x_{\\text{max}}-x_{\\text{min}}+1}}\\ \
    p^{x-x_{\\text{min}}} & x \\in \\{x_{\\text{min}},x_{\\text{min}}+1,\
    \\ldots,x_{\\text{max}}\\} \\\\ 0 & \\text{otherwise} \\end{cases},$$
    where \\(x_{\\text{min}}\\) is an integer and \\(x_{\\text{max}}\\) can be
    either an integer greater than or equal to \\(x_{\\text{min}}\\) or
    \\(\\infty\\).
    """
    def __init__(self, common_ratio, minimum=0, maximum=None, metadata=None):
        """
        Initializes a new `GeometricDistribution` with the given parameter
        values.
        
        Parameters
        ----------
        common_ratio : float
            real number, \\(p\\), in (0, 1)
        minimum : int
            minimum value that could be returned by this distribution
        maximum : int
            maximum value that could be returned by this distribution. if None,
            there is no maximum value
        metadata : number or str or dict or `distpy.util.Savable.Savable`
            data to store alongside this distribution.
        """
        self.common_ratio = common_ratio
        self.minimum = minimum
        self.maximum = maximum
        self.metadata = metadata
    
    @property
    def common_ratio(self):
        """
        The common ration between the probability mass function of successive
        integers.
        """
        if not hasattr(self, '_common_ratio'):
            raise AttributeError("common_ration was referenced before it " +\
                "was set.")
        return self._common_ratio
    
    @common_ratio.setter
    def common_ratio(self, value):
        """
        Setter for `GeometricDistribution.common_ratio`.
        
        Parameters
        ----------
        value : float
            a number between 0 and 1 (exclusive)
        """
        if type(value) in numerical_types:
            if (value > 0.) and (value < 1.):
                self._common_ratio = value
            else:
                raise ValueError("scale given to GeometricDistribution was " +\
                    "not between 0 and 1.")
        else:
            raise ValueError("common_ratio given to GeometricDistribution " +\
                "was not a number.")
    
    @property
    def minimum(self):
        """
        The minimum allowable value in this distribution.
        """
        if not hasattr(self, '_minimum'):
            raise AttributeError("minimum referenced before it was set.")
        return self._minimum
    
    @minimum.setter
    def minimum(self, value):
        """
        Setter for `GeometricDistribution.minimum`
        
        Parameters
        ----------
        value : int
            minimum value that can be returned by this `GeometricDistribution`
        """
        if type(value) in int_types:
            self._minimum = value
        else:
            raise TypeError("minimum was set to a non-int.")
    
    @property
    def maximum(self):
        """
        The maximum allowable value in this distribution.
        """
        if not hasattr(self, '_maximum'):
            raise AttributeError("maximum was referenced before it was set.")
        return self._maximum
    
    @maximum.setter
    def maximum(self, value):
        """
        Setter for `GeometricDistribution.maximum`
        
        Parameters
        ----------
        value : int or None
            - if None, there is no maximum and drawn values can be arbitrarily
            large
            - otherwise, `value` should be an integer greater than
            `GeometricDistribution.minimum`
        """
        if type(value) is type(None):
            self._maximum = None
        elif type(value) in int_types:
            if value >= self.minimum:
                self._maximum = value
            else:
                raise ValueError("maximum was not greater than minimum.")
        else:
            raise TypeError("maximum wasn't set to None or an integer.")
    
    @property
    def range(self):
        """
        One greater than the distance between `GeometricDistribution.minimum`
        and `GeometricDistribution.maximum`.
        """
        if not hasattr(self, '_range'):
            if type(self.maximum) is type(None):
                self._range = None
            else:
                self._range = (self.maximum - self.minimum + 1)
        return self._range
    
    @property
    def constant_in_log_value(self):
        """
        The portion of the log value which does not depend on the point at
        which the value is computed, given by \\(\\ln{(1-p)} - \\begin{cases}\
        \\ln{(1-p^r)} & r<\\infty \\\\ 0 & \\text{otherwise} \\end{cases}\\),
        where \\(r\\) is the range of values.
        """
        if not hasattr(self, '_constant_in_log_value'):
            self._constant_in_log_value = np.log(1 - self.common_ratio)
            if type(self.range) is not type(None):
                self._constant_in_log_value -=\
                    np.log(1 - (self.common_ratio ** self.range))
        return self._constant_in_log_value
    
    @property
    def numparams(self):
        """
        The number of parameters of this `GeometricDistribution`, 1.
        """
        return 1
    
    @property
    def mean(self):
        """
        The mean of this `GeometricDistribution`.
        """
        if not hasattr(self, '_mean'):
            mean = self.minimum + (self.common_ratio / (1 - self.common_ratio))
            if type(self.maximum) is not type(None):
                mean = mean -\
                    ((self.range * (self.common_ratio ** self.range)) /\
                    (1 - (self.common_ratio ** self.range)))
            self._mean = mean
        return self._mean
    
    @property
    def variance(self):
        """
        The variance of this `GeometricDistribution`.
        """
        if not hasattr(self, '_variance'):
            roomr = self.common_ratio / (1 - self.common_ratio)
            expected_square = roomr + (2 * (roomr ** 2))
            if type(self.maximum) is not type(None):
                rs = (self.common_ratio ** self.range)
                expected_square = expected_square - ((self.range ** 2) * rs) -\
                    (2 * roomr * self.range * rs) - (roomr * rs) -\
                    (2 * rs * (roomr ** 2))
                expected_square = expected_square /\
                    (1 - (self.common_ratio ** self.range))
            self._variance =\
                expected_square - ((self.mean - self.minimum) ** 2)
        return self._variance
    
    @property
    def log_common_ratio(self):
        """
        The natural logarithm of the common ratio of successive probabilities,
        given by \\(\\ln{p}\\).
        """
        if not hasattr(self, '_log_common_ratio'):
            self._log_common_ratio = np.log(self.common_ratio)
        return self._log_common_ratio
    
    def draw(self, shape=None, random=rand):
        """
        Draws point(s) from this `GeometricDistribution`.
        
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
        uniforms = random.uniform(size=shape)
        if type(self.maximum) is type(None):
            log_argument = uniforms
        else:
            log_argument =\
                (1 - (uniforms * (1 - (self.common_ratio ** self.range))))
        return self.minimum +\
            np.floor(np.log(log_argument) / self.log_common_ratio).astype(int)
    
    def log_value(self, point):
        """
        Computes the logarithm of the value of this `GeometricDistribution` at
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
            if point >= self.minimum:
                if (type(self.maximum) is not type(None)) and\
                    (point > self.maximum):
                    return -np.inf
                else:
                    return self.constant_in_log_value +\
                        ((point - self.minimum) * self.log_common_ratio)
            else:
                return -np.inf
        else:
            raise TypeError("point given to GeometricDistribution was not " +\
                "an integer.")

    def to_string(self):
        """
        Finds and returns a string version of this `GeometricDistribution` of
        the form `"Geometric(r)"`.
        """
        return "Geometric({:.4g})".format(self.common_ratio)
    
    def __eq__(self, other):
        """
        Checks for equality of this `GeometricDistribution` with `other`.
        
        Parameters
        ----------
        other : object
            object to check for equality
        
        Returns
        -------
        result : bool
            True if and only if `other` is a `GeometricDistribution` with the
            same `GeometricDistribution.minimum`,
            `GeometricDistribution.maximum` and
            `GeometricDistribution.common_ratio`
        """
        if isinstance(other, GeometricDistribution):
            ratios_close =\
                np.isclose(self.common_ratio, other.common_ratio, atol=1e-6)
            minima_equal = (self.minimum == other.minimum)
            maxima_equal = (self.maximum == other.maximum)
            metadata_equal = self.metadata_equal(other)
            return all([ratios_close, minima_equal, maxima_equal, metadata_equal])
        else:
            return False
    
    @property
    def can_give_confidence_intervals(self):
        """
        Discrete distributions do not support confidence intervals.
        """
        return False
    
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
        `GeometricDistribution` so that it can be loaded later.
        
        Parameters
        ----------
        group : h5py.Group
            hdf5 file group to fill
        save_metadata : bool
            - if True, attempts to save metadata alongside distribution and
            throws error if it fails
            - if False, metadata is ignored in saving process
        """
        group.attrs['class'] = 'GeometricDistribution'
        group.attrs['common_ratio'] = self.common_ratio
        group.attrs['minimum'] = self.minimum
        if type(self.maximum) is not type(None):
            group.attrs['maximum'] = self.maximum
        if save_metadata:
            self.save_metadata(group)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a `GeometricDistribution` from the given hdf5 file group.
        
        Parameters
        ----------
        group : h5py.Group
            the same hdf5 file group which fill_hdf5_group was called on when
            this Distribution was saved
        
        Returns
        -------
        distribution : `GeometricDistribution`
            distribution created from the information in the given group
        """
        try:
            assert group.attrs['class'] == 'GeometricDistribution'
        except:
            raise TypeError("The given hdf5 file doesn't seem to contain a " +\
                "GeometricDistribution.")
        metadata = Distribution.load_metadata(group)
        common_ratio = group.attrs['common_ratio']
        minimum = group.attrs['minimum']
        if 'maximum' in group.attrs:
            maximum = group.attrs['maximum']
        else:
            maximum = None
        return GeometricDistribution(common_ratio, minimum=minimum,\
            maximum=maximum, metadata=metadata)
    
    @property
    def gradient_computable(self):
        """
        Boolean describing whether the gradient of the given distribution has
        been implemented. If True,
        `GeometricDistribution.gradient_of_log_value` method can be called
        safely.
        """
        return False 
    
    @property
    def hessian_computable(self):
        """
        Boolean describing whether the hessian of the given distribution has
        been implemented. If True,
        `GeometricDistribution.hessian_of_log_value` method can be called
        safely.
        """
        return False
    
    def copy(self):
        """
        Copies this distribution.
        
        Returns
        -------
        copied : `GeometricDistribution`
            a deep copy of this distribution, ignoring metadata.
        """
        return GeometricDistribution(self.common_ratio, self.minimum,\
            self.maximum)

