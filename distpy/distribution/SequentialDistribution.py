"""
Module containing class representing a distribution whose drawn values are
sorted draws from a univariate distribution. Its PDF is represented by:
$$f(x_1,x_2,\\ldots,x_N) = \\begin{cases} N!\\ \\prod_{k=1}^Ng(x_k) &\
x_1\\le x_2 \\le \\ldots \\le x_N \\\\ 0 & \\text{otherwise} \\end{cases},$$
where \\(g\\) is the PDF (or PMF) of any univariate distribution.

**File**: $DISTPY/distpy/distribution/SequentialDistribution.py  
**Author**: Keith Tauscher  
**Date**: 31 May 2021
"""
import numpy as np
import numpy.random as rand
from scipy.special import gammaln as log_gamma
from ..util import int_types, numerical_types, sequence_types
from .Distribution import Distribution

class SequentialDistribution(Distribution):
    """
    Class representing a distribution whose drawn values are sorted draws from
    a univariate distribution. Its PDF is represented by:
    $$f(x_1,x_2,\\ldots,x_N) = \\begin{cases} N!\\ \\prod_{k=1}^Ng(x_k) &\
    x_1\\le x_2 \\le \\ldots \\le x_N \\\\ 0 & \\text{otherwise}\
    \\end{cases},$$ where \\(g\\) is the PDF (or PMF) of any univariate
    distribution.
    """
    def __init__(self, shared_distribution, numparams=2, metadata=None):
        """
        Initializes a new `SequentialDistribution` with the given parameter
        values.
        
        Parameters
        ----------
        shared_distribution : `distpy.distribution.Distribution.Distribution`
            distribution, with PDF \\(g\\), shared by the points before sorting
        numparams : int
            integer number of parameters this distribution describes
        metadata : number or str or dict or `distpy.util.Savable.Savable`
            data to store alongside this distribution.
        """
        self.shared_distribution = shared_distribution
        self.numparams = numparams
        self.metadata = metadata
    
    @property
    def shared_distribution(self):
        """
        The distribution from which to draw values before sorting.
        """
        if not hasattr(self, '_shared_distribution'):
            raise AttributeError("shared_distribution was referenced " +\
                "before it was set.")
        return self._shared_distribution
    
    @shared_distribution.setter
    def shared_distribution(self, value):
        """
        Setter for `SequentialDistribution.shared_distribution`.
        
        Parameters
        ----------
        value : `distpy.distribution.Distribution.Distribution`
            univariate distribution
        """
        if isinstance(value, Distribution):
            if value.numparams == 1:
                self._shared_distribution = value
            else:
                raise NotImplementedError("The shared_distribution " +\
                    "provided to a SequentialDistribution was multivariate " +\
                    "(I don't know how to deal with this!).")
        else:
            raise ValueError("The shared_distribution given to a " +\
                "SequentialDistribution was not recognizable as a " +\
                "distribution.")
    
    @property
    def numparams(self):
        """
        The number of parameters of this `SequentialDistribution`.
        """
        if not hasattr(self, '_numparams'):
            raise AttributeError("numparams was referenced before it was set.")
        return self._numparams
    
    @property
    def mean(self):
        """
        The mean of the `SequentialDistribution` class is not implemented.
        """
        if not hasattr(self, '_mean'):
            raise AttributeError("mean is not implemented for the " +\
                "SequentialDistribution class.")
        return self._mean
    
    @property
    def variance(self):
        """
        The variance of the `SequentialDistribution` class is not implemented.
        """
        if not hasattr(self, '_variance'):
            raise AttributeError("variance is not implemented for the " +\
                "SequentialDistribution class.")
        return self._variance
    
    @numparams.setter
    def numparams(self, value):
        """
        Setter for `SequentialDistribution.numparams`.
        
        Parameters
        ----------
        value : int
            a positive integer greater than 1.
        """
        if (type(value) in int_types):
            if int(value) >= 1:
                self._numparams = int(value)
            else:
                raise ValueError("A SequentialDistribution was initialized " +\
                    "with non-positive numparams.")
        else:
            raise ValueError("The type of the number of parameters given " +\
                "to a SequentialDistribution was not numerical.")
    
    def draw(self, shape=None, random=rand):
        """
        Draws point(s) from this `SequentialDistribution`. Below, `p` is
        `SequentialDistribution.numparams`.
        
        Parameters
        ----------
        shape : int or tuple or None
            - if None, returns single random variate as a 1D array of length
            `p` is returned
            - if int, \\(n\\), returns \\(n\\) random variates as a 2D
            array of shape `(n,p)` is returned
            - if tuple of \\(n\\) ints, returns `numpy.prod(shape)` random
            variates as an \\((n+1)\\)-D array of shape `shape+(p,)` is
            returned
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
            shape = (1,)
        elif type(shape) in int_types:
            shape = (shape,)
        unsorted =\
            self.shared_distribution.draw(shape=shape+(self.numparams,),\
            random=random)
        points = np.sort(np.array(unsorted), axis=-1)
        if self.numparams == 1:
            points = points[...,0]
        if none_shape:
            return points[0]
        else:
            return points

    def log_value(self, point):
        """
        Computes the logarithm of the value of this `SequentialDistribution` at
        the given point.
        
        Parameters
        ----------
        point : `numpy.ndarray`
            if this distribution describes \\(p\\) parameters, `point` should
            be a length-\\(p\\) `numpy.ndarray`
        
        Returns
        -------
        value : float
            natural logarithm of the value of this distribution at `point`. If
            \\(f\\) is this distribution's PDF and \\(x\\) is `point`, then
            `value` is \\(\\ln{\\big(f(x)\\big)}\\)
        """
        if (type(point) in numerical_types) and (self.numparams == 1):
            result = self.shared_distribution.log_value(point)
        elif type(point) in sequence_types:
            if len(point) == self.numparams:
                if np.all(point[1:] >= point[:-1]):
                    result = log_gamma(self.numparams + 1)
                    for ipar in range(self.numparams):
                        result +=\
                            self.shared_distribution.log_value(point[ipar])
                else:
                    return -np.inf
            else:
                raise ValueError("The length of the point provided to a " +\
                    "SequentialDistribution was not the same as the " +\
                    "SequentialDistribution's number of parameters")
        else:
            raise ValueError("The point given to a SequentialDistribution " +\
                "was not of a list type.")
        return result

    def to_string(self):
        """
        Finds and returns a string version of this `SequentialDistribution` of
        the form `"Sequential(shared)"`, where `"shared"` is the string
        representation of `SequentialDistribution.shared_distribution`.
        """
        return "Sequential({!s})".format(self.shared_distribution.to_string())
    
    def __eq__(self, other):
        """
        Checks for equality of this `SequentialDistribution` with `other`.
        
        Parameters
        ----------
        other : object
            object to check for equality
        
        Returns
        -------
        result : bool
            True if and only if `other` is a `SequentialDistribution` with the
            same `SequentialDistribution.shared_distribution` and
            `SequentialDistribution.numparams`
        """
        if isinstance(other, SequentialDistribution):
            numparams_equal = (self.numparams == other.numparams)
            shared_distribution_equal =\
                (self.shared_distribution == other.shared_distribution)
            metadata_equal = self.metadata_equal(other)
            return all([numparams_equal, shared_distribution_equal,\
                metadata_equal])
        else:
            return False
    
    @property
    def minimum(self):
        """
        The minimum allowable value(s) in this distribution.
        """
        return [self.shared_distribution.minimum] * self.numparams
    
    @property
    def maximum(self):
        """
        The maximum allowable value(s) in this distribution.
        """
        return [self.shared_distribution.maximum] * self.numparams
    
    @property
    def is_discrete(self):
        """
        Boolean describing whether this distribution is discrete (True) or
        continuous (False).
        """
        return self.shared_distribution.is_discrete
    
    def fill_hdf5_group(self, group, save_metadata=True):
        """
        Fills the given hdf5 file group with data about this
        `SequentialDistribution` so that it can be loaded later.
        
        Parameters
        ----------
        group : h5py.Group
            hdf5 file group to fill
        save_metadata : bool
            - if True, attempts to save metadata alongside distribution and
            throws error if it fails
            - if False, metadata is ignored in saving process
        """
        group.attrs['class'] = 'SequentialDistribution'
        group.attrs['numparams'] = self.numparams
        subgroup = group.create_group('shared_distribution')
        self.shared_distribution.fill_hdf5_group(subgroup)
        if save_metadata:
            self.save_metadata(group)
    
    @staticmethod
    def load_from_hdf5_group(group, shared_distribution_class, *args,\
        **kwargs):
        """
        Loads a `SequentialDistribution` from the given hdf5 file group.
        
        Parameters
        ----------
        group : h5py.Group
            the same hdf5 file group which fill_hdf5_group was called on when
            this Distribution was saved
        shared_distribution_class : class
            the class of the distribution shared by the variables
        args : sequence
            positional arguments to pass to `load_from_hdf5_group` method of
            `shared_distribution_class`
        kwargs : dict
            keyword arguments to pass to `load_from_hdf5_group` method of
            `shared_distribution_class`
        
        Returns
        -------
        distribution : `SequentialDistribution`
            distribution created from the information in the given group
        """
        try:
            assert group.attrs['class'] == 'SequentialDistribution'
        except:
            raise TypeError("The given hdf5 file doesn't seem to contain a " +\
                "SequentialDistribution.")
        metadata = Distribution.load_metadata(group)
        numparams = group.attrs['numparams']
        shared_distribution = shared_distribution_class.load_from_hdf5_group(\
            group['shared_distribution'], *args, **kwargs)
        return SequentialDistribution(shared_distribution=shared_distribution,\
            numparams=numparams, metadata=metadata)
    
    @property
    def gradient_computable(self):
        """
        Boolean describing whether the gradient of the given distribution has
        been implemented. If True,
        `SequentialDistribution.gradient_of_log_value` method can be called
        safely.
        """
        return self.shared_distribution.gradient_computable
    
    def gradient_of_log_value(self, point):
        """
        Computes the gradient (derivative) of the logarithm of the value of
        this `SequentialDistribution` at the given point.
        
        Parameters
        ----------
        point : `numpy.ndarray`
            if this distribution describes \\(p\\) parameters, `point` should
            be a length-\\(p\\) `numpy.ndarray`
        
        Returns
        -------
        value : `numpy.ndarray`
            gradient of the natural logarithm of the value of this
            distribution. If \\(f\\) is this distribution's PDF and \\(x\\) is
            `point`, then `value` is
            \\(\\boldsymbol{\\nabla}\\ln{\\big(f(x)\\big)}\\) as a 1D
            `numpy.ndarray` of length \\(p\\)
        """
        if self.numparams == 1:
            return self.shared_distribution.gradient_of_log_value(point)
        elif np.all(point[1:] >= point[:-1]):
            answer = []
            for parameter in point:
                answer.append(\
                    self.shared_distribution.gradient_of_log_value(parameter))
            return np.array(answer)
        else:
            return np.zeros((self.numparams,))
    
    @property
    def hessian_computable(self):
        """
        Boolean describing whether the hessian of the given distribution has
        been implemented. If True,
        `SequentialDistribution.hessian_of_log_value` method can be called
        safely.
        """
        return self.shared_distribution.hessian_computable
    
    def hessian_of_log_value(self, point):
        """
        Computes the hessian (second derivative) of the logarithm of the value
        of this `SequentialDistribution` at the given point.
        
        Parameters
        ----------
        point : `numpy.ndarray`
            if this distribution describes \\(p\\) parameters, `point` should
            be a length-\\(p\\) `numpy.ndarray`
        
        Returns
        -------
        value : `numpy.ndarray`
            hessian of the natural logarithm of the value of this
            distribution. If \\(f\\) is this distribution's PDF and \\(x\\) is
            `point`, then `value` is \\(\\boldsymbol{\\nabla}\
            \\boldsymbol{\\nabla}^T\\ln{\\big(f(x)\\big)}\\) as a 2D
            `numpy.ndarray` that is \\(p\\times p\\)
        """
        if self.numparams == 1:
            return self.shared_distribution.hessian_of_log_value(point)
        elif np.all(point[1:] >= point[:-1]):
            answer = []
            for parameter in point:
                answer.append(\
                    self.shared_distribution.hessian_of_log_value(parameter))
            return np.diag(answer)
        else:
            return np.zeros((self.numparams,) * 2)
    
    def copy(self):
        """
        Copies this distribution.
        
        Returns
        -------
        copied : `SequentialDistribution`
            a deep copy of this distribution, ignoring metadata.
        """
        return SequentialDistribution(self.shared_distribution.copy(),\
            self.numparams)

