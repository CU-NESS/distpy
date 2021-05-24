"""
Module containing class representing Gaussian distribution with a block
diagonal covariance matrix, \\(\\begin{bmatrix} \\boldsymbol{\\Sigma}_1 &\
\\boldsymbol{0} & \\cdots & \\boldsymbol{0} \\\\ \\boldsymbol{0} &\
\\boldsymbol{\\Sigma}_2 & \\cdots & \\boldsymbol{0} \\\\ \\vdots & \\vdots &\
\\ddots & \\vdots \\\\ \\boldsymbol{0} & \\boldsymbol{0} & \\cdots &\
\\boldsymbol{\\Sigma}_N \\end{bmatrix},\\) represented by a
`distpy.util.SparseSquareBlockDiagonalMatrix.SparseSquareBlockDiagonalMatrix`
object.

**File**: $DISTPY/distpy/distribution/SparseGaussianDistribution.py  
**Author**: Keith Tauscher  
**Date**: 22 May 2021
"""
from __future__ import division
import numpy as np
import numpy.random as rand
from ..util import real_numerical_types, int_types, sequence_types,\
    create_hdf5_dataset, get_hdf5_value, SparseSquareBlockDiagonalMatrix
from .Distribution import Distribution
from .GaussianDistribution import GaussianDistribution

natural_log_two_pi = np.log(2 * np.pi)
try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str

class SparseGaussianDistribution(Distribution):
    """
    Class representing Gaussian distribution with a block diagonal covariance
    matrix, \\(\\begin{bmatrix} \\boldsymbol{\\Sigma}_1 & \\boldsymbol{0} &\
    \\cdots & \\boldsymbol{0} \\\\ \\boldsymbol{0} & \\boldsymbol{\\Sigma}_2 &\
    \\cdots & \\boldsymbol{0} \\\\ \\vdots & \\vdots & \\ddots & \\vdots \\\\\
    \\boldsymbol{0} & \\boldsymbol{0} & \\cdots & \\boldsymbol{\\Sigma}_N\
    \\end{bmatrix},\\) represented by a
    `distpy.util.SparseSquareBlockDiagonalMatrix.SparseSquareBlockDiagonalMatrix`
    object.
    """
    def __init__(self, mean, covariance, metadata=None):
        """
        Initializes a new `SparseGaussianDistribution`.
        
        Parameters
        ----------
        mean : numpy.ndarray
            1D array of numbers, \\(\\boldsymbol{\\mu}\\)
        covariance : `distpy.util.SparseSquareBlockDiagonalMatrix.SparseSquareBlockDiagonalMatrix`
            covariance matrix of the distribution, \\(\\boldsymbol{\\Sigma}\\)
        metadata : object
            extra data to store alongside the distribution. See
            `distpy.distribution.Distribution.Distribution.metadata` for more
            details
        """
        self.mean = mean
        self.covariance = covariance
        self.metadata = metadata
        # this will allow for __rmatmul__ to implement left matrix
        # multiplication in the future, when numpy adds __matmul__ as a ufunc
        # (in the future).
        self.__array_ufunc__ = None
    
    @staticmethod
    def combine(*distributions):
        """
        Combines many `SparseGaussianDistribution` objects into one by
        concatenating their means and covariance matrices.
        
        Parameters
        ----------
        distributions : sequence
            a sequence of `SparseGaussianDistribution` objects to combine
        
        returns: a single `SparseGaussianDistribution` object 
        """
        if all([isinstance(distribution, SparseGaussianDistribution)\
            for distribution in distributions]):
            new_mean = np.concatenate([distribution.mean\
                for distribution in distributions])
            new_covariance = SparseSquareBlockDiagonalMatrix.concatenate(\
                *[distribution.covariance for distribution in distributions])
            return SparseGaussianDistribution(new_mean, new_covariance)
        else:
            raise TypeError("At least one of the distributions given to " +\
                "the SparseGaussianDistribution class' combine function " +\
                "was not a SparseGaussianDistribution.")
    
    @property
    def mean(self):
        """
        The mean, \\(\\boldsymbol{\\mu}\\), of this
        `SparseGaussianDistribution` in `numpy.ndarray` form.
        """
        if not hasattr(self, '_mean'):
            raise AttributeError("mean was referenced before it was set.")
        return self._mean
    
    @mean.setter
    def mean(self, value):
        """
        Setter for `SparseGaussianDistribution.mean`
        
        Parameters
        ----------
        value : sequence
            1D sequence of numbers
        """
        if type(value) in real_numerical_types:
            value = [value]
        if type(value) in sequence_types:
            value = np.array(value)
            if value.ndim != 1:
                raise ValueError("The mean of a SparseGaussianDistribution " +\
                    "was not 1 dimensional.")
            elif value.size == 0:
                raise ValueError("The mean of a SparseGaussianDistribution " +\
                    "was set to something like an empty array.")
            elif value.size == 1:
                raise ValueError("The mean of a SparseGaussianDistribution " +\
                    "was given as a single number or sequence of length 1. " +\
                    "This would imply that this distribution should be " +\
                    "univariate. However, the SparseGaussianDistribution " +\
                    "class is meant to for large numbers of parameters. " +\
                    "The standard GaussianDistribution class should be " +\
                    "used for a univariate Gaussian distribution.")
            else:
                self._mean = value
        else:
            raise ValueError("The mean of a SparseGaussianDistribution was " +\
                "neither a number nor a sequence of numbers.")
   
    @property
    def numparams(self):
        """
        The number of parameters which this `SparseGaussianDistribution`
        describes (same as dimension of mean and covariance).
        """
        if not hasattr(self, '_numparams'):
            self._numparams = len(self.mean)
        return self._numparams
    
    @property
    def covariance(self):
        """
        The covariance matrix, \\(\\boldsymbol{\\Sigma}\\), of this Gaussian as
        a
        `distpy.util.SparseSquareBlockDiagonalMatrix.SparseSquareBlockDiagonalMatrix`.
        """
        if not hasattr(self, '_covariance'):
            raise AttributeError("covariance was referenced before it was " +\
                "set.")
        return self._covariance
    
    @covariance.setter
    def covariance(self, value):
        """
        Setter for the `SparseGaussianDistribution.covariance` property that
        stores the covariance matrix of this Gaussian distribution.
        
        value : `distpy.util.SparseSquareBlockDiagonalMatrix.SparseSquareBlockDiagonalMatrix`
            `distpy.util.SparseSquareBlockDiagonalMatrix.SparseSquareBlockDiagonalMatrix`
            object representing the full covariance matrix
        """
        if isinstance(value, SparseSquareBlockDiagonalMatrix):
            if value.dimension == self.numparams:
                if value.positive_definite:
                    self._covariance = value
                else:
                    raise ValueError("The covariance given to a " +\
                        "SparseGaussianDistribution was not " +\
                        "positive-definite.")
            else:
                raise ValueError(("The dimension of the covariance matrix " +\
                    "({0:d}) was not equal to the dimension of the mean " +\
                    "vector ({1:d}).").format(value.dimension, self.numparams))
        else:
            raise TypeError("covariance was set to a " +\
                "non-SparseSquareBlockDiagonalMatrix object.")
    
    @property
    def log_determinant_covariance(self):
        """
        The natural logarithm of the determinant of the covariance matrix,
        \\(\\ln{\\Vert\\boldsymbol{\\Sigma}\\Vert}\\).
        """
        if not hasattr(self, '_log_determinant_covariance'):
            self._log_determinant_covariance =\
                self.covariance.sign_and_log_abs_determinant()[1]
        return self._log_determinant_covariance
    
    @property
    def inverse_covariance(self):
        """
        The inverse covariance, \\(\\boldsymbol{\\Sigma}^{-1}\\), as a
        `distpy.util.SparseSquareBlockDiagonalMatrix.SparseSquareBlockDiagonalMatrix`.
        """
        if not hasattr(self, '_inverse_covariance'):
            self._inverse_covariance = self.covariance.inverse()
        return self._inverse_covariance
    
    @property
    def square_root_covariance(self):
        """
        The square root covariance, \\(\\boldsymbol{\\Sigma}^{1/2}\\), as a
        `distpy.util.SparseSquareBlockDiagonalMatrix.SparseSquareBlockDiagonalMatrix`.
        """
        if not hasattr(self, '_square_root_covariance'):
            self._square_root_covariance = self.covariance.square_root()
        return self._square_root_covariance
    
    @property
    def inverse_square_root_covariance(self):
        """
        The inverse square root covariance, \\(\\boldsymbol{\\Sigma}^{-1/2}\\),
        as a
        `distpy.util.SparseSquareBlockDiagonalMatrix.SparseSquareBlockDiagonalMatrix`.
        """
        if not hasattr(self, '_inverse_square_root_covariance'):
            self._inverse_square_root_covariance =\
                self.covariance.inverse_square_root()
        return self._inverse_square_root_covariance
    
    def weight(self, array, axis=0):
        """
        Weights the given array by the inverse square root of the covariance
        matrix of this distribution.
        
        Parameters
        ----------
        array : numpy.ndarray
            the array to weight, can be any number of dimensions as long as the
            specified one has length given by
            `SparseGaussianDistribution.numparams`
        axis : int
            index of the axis corresponding to the parameters
        
        Returns
        -------
        weighted : numpy.ndarray
            array of same shape as array corresponding to
            \\(\\boldsymbol{\\Sigma}^{-1/2}\\boldsymbol{A}\\), where
            \\(\\boldsymbol{A}\\) is array shaped so that the matrix
            multiplication makes sense.
        """
        axis = axis % array.ndim
        if array.ndim == 1:
            return self.inverse_square_root_covariance.__matmul__(array)
        elif array.ndim == 2:
            if axis == 0:
                return\
                    self.inverse_square_root_covariance.__matmul__(array.T).T
            else:
                return self.inverse_square_root_covariance.__matmul__(array)
        else:
            before_shape = array.shape[:axis]
            after_shape = array.shape[(axis+1):]
            if axis == (array.ndim - 1):
                weighted_array = array
            else:
                weighted_array = np.rollaxis(array, axis, start=array.ndim)
            weighted_array =\
                self.inverse_square_root_covariance.__matmul__(weighted_array)
            if axis != (array.ndim - 1):
                weighted_array =\
                    np.rollaxis(weighted_array, array.ndim - 1, start=axis)
            return weighted_array
    
    @property
    def variance(self):
        """
        Dense matrix form of the covariance of this distribution as a
        `numpy.ndarray`.
        """
        if not hasattr(self, '_variance'):
            self._variance = self.covariance.dense()
        return self._variance
    
    def dense(self):
        """
        Finds a standard
        `distpy.distribution.GaussianDistribution.GaussianDistribution` object
        representing the same random variate as this distribution does. This
        involves creating a dense version of the covariance matrix, which may
        be prohibitive if the dimension is large enough compared to system
        memory.
        
        Returns
        -------
        dist : `distpy.distribution.GaussianDistribution.GaussianDistribution`
            a dense form of this distribution
        """
        return GaussianDistribution(self.mean, self.variance,\
            metadata=self.metadata)
    
    def __add__(self, other):
        """
        Adds the Gaussian random variate described by this distribution
        to the given object.
        
        Parameters
        ----------
        other : number or `numpy.ndarray` or `SparseGaussianDistribution`
            - if `other` is a constant, the returned Gaussian is the same as
            this one with `other` added to the mean and the same covariance
            - if `other` is a 1D numpy.ndarray, \\(\\boldsymbol{a}\\), it must
            be of the same length as the dimension of this
            `SparseGaussianDistribution`. In this case, the returned
            `SparseGaussianDistribution` is the distribution of the sum of this
            Gaussian variate with other, which has a translated mean,
            \\(\\boldsymbol{\\mu}^\\prime=\\boldsymbol{\\mu}+\
            \\boldsymbol{a}\\), but the same covariance,
            \\(\\boldsymbol{\\Sigma}\\)
            - if `other` is a `SparseGaussianDistribution` with mean
            \\(\\boldsymbol{\\nu}\\) and covariance
            \\(\\boldsymbol{\\Lambda}\\), the returned distribution has mean,
            \\(\\boldsymbol{\\mu}+\\boldsymbol{\\nu}\\), and covariance,
            \\(\\boldsymbol{\\Sigma}+\\boldsymbol{\\Lambda}\\)
        
        Returns
        -------
        sum : `SparseGaussianDistribution`
            distribution of sum of the random variate described by this
            distribution and `other`. See documentation of `other` above for
            details.
        """
        if isinstance(other, SparseGaussianDistribution):
            if self.numparams == other.numparams:
                new_mean = self.mean + other.mean
                new_covariance = self.covariance + other.covariance
            else:
                raise ValueError("Cannot add together two " +\
                    "SparseGaussianDistribution objects with different " +\
                    "numbers of parameters.")
        elif type(other) in [list, tuple, np.ndarray]:
            other = np.array(other)
            if other.ndim == 1:
                if len(other) == self.numparams:
                    new_mean = self.mean + other
                    new_covariance = self.covariance
                else:
                    raise ValueError("Cannot multiply Gaussian distributed " +\
                        "random vector by a vector of different size.")
            else:
                raise ValueError("Cannot multiply Gaussian distributed " +\
                    "random vector by a tensor with more than 1 index.")
        else:
            # assume other is a constant
            new_mean = self.mean + other
            new_covariance = self.covariance
        return SparseGaussianDistribution(new_mean, new_covariance)
    
    def __radd__(self, other):
        """
        Adds the Gaussian random variate described by this distribution
        to the given object. Equivalent to `SparseGaussianDistribution.__add__`
        method (making addition commutative).
        
        Parameters
        ----------
        other : number or `numpy.ndarray` or `SparseGaussianDistribution`
            - if `other` is a constant, the returned Gaussian is the same as
            this one with `other` added to the mean and the same covariance
            - if `other` is a 1D numpy.ndarray, \\(\\boldsymbol{a}\\), it must
            be of the same length as the dimension of this
            `SparseGaussianDistribution`. In this case, the returned
            `SparseGaussianDistribution` is the distribution of the sum of this
            Gaussian variate with other, which has a translated mean,
            \\(\\boldsymbol{\\mu}^\\prime=\\boldsymbol{\\mu}+\
            \\boldsymbol{a}\\), but the same covariance,
            \\(\\boldsymbol{\\Sigma}\\)
            - if `other` is a `SparseGaussianDistribution` with mean
            \\(\\boldsymbol{\\nu}\\) and covariance
            \\(\\boldsymbol{\\Lambda}\\), the returned distribution has mean,
            \\(\\boldsymbol{\\mu}+\\boldsymbol{\\nu}\\), and covariance,
            \\(\\boldsymbol{\\Sigma}+\\boldsymbol{\\Lambda}\\)
        
        Returns
        -------
        sum : `SparseGaussianDistribution`
            distribution of sum of the random variate described by this
            distribution and `other`. See documentation of `other` above for
            details.
        """
        return self.__add__(other)
    
    def __sub__(self, other):
        """
        Adds the Gaussian random variate described by this distribution
        to the negative of the given object.
        
        Parameters
        ----------
        other : number or `numpy.ndarray` or `SparseGaussianDistribution`
            - if `other` is a constant, the returned Gaussian is the same as
            this one with `other` subtracted from the mean and the same
            covariance
            - if `other` is a 1D numpy.ndarray, \\(\\boldsymbol{a}\\), it must
            be of the same length as the dimension of this
            `SparseGaussianDistribution`. In this case, the returned
            `SparseGaussianDistribution` is the distribution of the sum of this
            Gaussian variate with other, which has a translated mean,
            \\(\\boldsymbol{\\mu}^\\prime=\\boldsymbol{\\mu}-\
            \\boldsymbol{a}\\), but the same covariance,
            \\(\\boldsymbol{\\Sigma}\\)
            - if `other` is a `SparseGaussianDistribution` with mean
            \\(\\boldsymbol{\\nu}\\) and covariance
            \\(\\boldsymbol{\\Lambda}\\), the returned distribution has mean,
            \\(\\boldsymbol{\\mu}-\\boldsymbol{\\nu}\\), and covariance,
            \\(\\boldsymbol{\\Sigma}+\\boldsymbol{\\Lambda}\\)
        
        Returns
        -------
        sum : `SparseGaussianDistribution`
            distribution of difference of the random variate described by this
            distribution and `other`. See documentation of `other` above for
            details.
        """
        return self.__add__(-other)
    
    def __rsub__(self, other):
        """
        Adds the Gaussian random variate described by this distribution
        to the negative of the given object. Equivalent to
        `SparseGaussianDistribution.__sub__(other).__neg__()`.
        
        Parameters
        ----------
        other : number or `numpy.ndarray` or `SparseGaussianDistribution`
            - if `other` is a constant, the returned Gaussian is the same as
            this one with `other` subtracted from the mean and the same
            covariance
            - if `other` is a 1D numpy.ndarray, \\(\\boldsymbol{a}\\), it must
            be of the same length as the dimension of this
            `SparseGaussianDistribution`. In this case, the returned
            `SparseGaussianDistribution` is the distribution of the sum of this
            Gaussian variate with other, which has a translated mean,
            \\(\\boldsymbol{\\mu}^\\prime=\\boldsymbol{\\mu}-\
            \\boldsymbol{a}\\), but the same covariance,
            \\(\\boldsymbol{\\Sigma}\\)
            - if `other` is a `SparseGaussianDistribution` with mean
            \\(\\boldsymbol{\\nu}\\) and covariance
            \\(\\boldsymbol{\\Lambda}\\), the returned distribution has mean,
            \\(\\boldsymbol{\\mu}-\\boldsymbol{\\nu}\\), and covariance,
            \\(\\boldsymbol{\\Sigma}+\\boldsymbol{\\Lambda}\\)
        
        Returns
        -------
        sum : `SparseGaussianDistribution`
            distribution of difference of the random variate described by this
            distribution and `other`. See documentation of `other` above for
            details.
        """
        return self.__sub__(other).__neg__()
    
    def __neg__(self):
        """
        Finds the distribution of the negative of the random variable this
        distribution describes.
        
        Returns
        -------
        result : `SparseGaussianDistribution`
            distribution with mean, \\(-\\boldsymbol{\\mu}\\), and covariance,
            \\(\\boldsymbol{\\Sigma}\\)
        """
        return SparseGaussianDistribution(-self.mean, self.covariance_list)
    
    def __mul__(self, other):
        """
        Multiplies the Gaussian random variate described by this distribution
        by the given constant.
        
        Parameters
        ----------
        other : number
            any non-zero scalar, \\(a\\)
        
        returns: `SparseGaussianDistribution`
            distribution of the product of the random variable with this
            distribution and the constant `other`, which has mean
            \\(a\\boldsymbol{\\mu}\\) and covariance
            \\(a^2\\boldsymbol{\\Sigma}\\)
        """
        new_mean = self.mean * other
        new_covariance = self.covariance * (other ** 2)
        return SparseGaussianDistribution(new_mean, new_covariance)
    
    def __rmul__(self, other):
        """
        Multiplies the Gaussian random variate described by this distribution
        by the given constant. Equivalent to
        `SparseGaussianDistribution.__mul__` method so that multiplication is
        commutative.
        
        Parameters
        ----------
        other : number
            any non-zero scalar, \\(a\\)
        
        returns: `SparseGaussianDistribution`
            distribution of the product of the random variable with this
            distribution and the constant `other`, which has mean
            \\(a\\boldsymbol{\\mu}\\) and covariance
            \\(a^2\\boldsymbol{\\Sigma}\\)
        """
        return self.__mul__(other)
    
    def __div__(self, other):
        """
        Divides the Gaussian random variate described by this distribution
        by the given constant.
        
        Parameters
        ----------
        other : number
            any non-zero scalar, \\(a\\)
        
        returns: `SparseGaussianDistribution`
            distribution of the product of the random variable with this
            distribution and the constant `other`, which has mean
            \\(\\boldsymbol{\\mu}/a\\) and covariance
            \\(\\boldsymbol{\\Sigma}/a^2\\)
        """
        if other == 0:
            raise ZeroDivisionError("Cannot divide by zero.")
        else:
            return self.__mul__(1 / other)
    
    def __matmul__(self, other):
        """
        Finds and returns the distribution of a matrix product of this random
        variate.
        
        Parameters
        ----------
        other : numpy.ndarray or\
        `distpy.util.SparseSquareBlockDiagonalMatrix.SparseSquareBlockDiagonalMatrix`
            - if `other` is a 1D numpy.ndarray, \\(\\boldsymbol{v}\\), it must
            have dimension `SparseGaussianDistribution.numparams`. In this
            case, the returned distribution is the distribution of
            \\(\\boldsymbol{v}\\cdot\\boldsymbol{x}\\), where
            \\(\\boldsymbol{x}\\) is the random variate described by this
            distribution. In particular, it returns a
            `distpy.distribution.GaussianDistribution.GaussianDistribution`
            with mean, \\(\\boldsymbol{v}\\cdot\\boldsymbol{\\mu}\\), and
            covariance,
            \\(\\boldsymbol{v}^T\\boldsymbol{\\Sigma}\\boldsymbol{v}\\)
            - if `other` is a 2D numpy.ndarray, \\(\\boldsymbol{A}\\), it must
            have shape `(m, n)`, where \\(n\\) is
            `SparseGaussianDistribution.numparams` and \\(m\\le n\\). The
            returned distribution is the distribution of
            `other` (matrix) multiplied with this Gaussian variate,
            \\(\\boldsymbol{x}\\), i.e. a
            `distpy.distribution.GaussianDistribution.GaussianDistribution`
            with mean, \\(\\boldsymbol{A}\\boldsymbol{x}\\), and covariance,
            \\(\\boldsymbol{A}\\boldsymbol{\\Sigma}\\boldsymbol{A}^T\\). Note
            that \\(\\boldsymbol{A}\\) must be full-rank (i.e. its rank must be
            \\(m\\)).
            - if `other` is a
            `distpy.util.SparseSquareBlockDiagonalMatrix.SparseSquareBlockDiagonalMatrix`,
            \\(\\boldsymbol{A}\\), it will return the same distribution given
            above in the case of a 2D array `other`, except it will be returned
            in a `SparseGaussianDistribution` instead of a
            `distpy.distribution.GaussianDistribution.GaussianDistribution`
        
        Returns
        -------
        distribution_of_product : `SparseGaussianDistribution` or\
        `distpy.distribution.GaussianDistribution.GaussianDistribution`
            the distribution of the product of this
            `SparseGaussianDistribution` and `other`. See documentation for
            `other` for details on the returned value for different inputs
        """
        if type(other) in sequence_types:
            other = np.array(other)
            if other.ndim == 1:
                if len(other) == self.numparams:
                    new_mean = np.vdot(self.mean, other)
                    new_covariance =\
                        np.vdot(self.covariance.__matmul__(other), other)
                else:
                    raise ValueError("Cannot multiply Gaussian distributed " +\
                        "random vector by a vector of different size.")
            elif other.ndim == 2:
                if other.shape[1] == self.numparams:
                    if other.shape[0] <= self.numparams:
                        # other is a matrix with self.numparams columns
                        new_mean = np.matmul(other, self.mean)
                        new_covariance = np.matmul(\
                            self.covariance.array_matrix_multiplication(other,\
                            right=False), other.T)
                    else:
                        raise ValueError("Cannot multiply Gaussian " +\
                            "distributed random vector by matrix which " +\
                            "will expand the number of parameters because " +\
                            "the covariance matrix of the result would be " +\
                            "singular.")
                else:
                    raise ValueError("Cannot multiply given matrix with " +\
                        "Gaussian distributed random vector because the " +\
                        "axis of its second dimension is not the same " +\
                        "length as the random vector.")
            else:
                raise ValueError("Cannot multiply Gaussian distributed " +\
                    "random vector by a tensor with more than 2 indices.")
        elif isinstance(other, SparseSquareBlockDiagonalMatrix):
            new_mean = other.__matmul__(self.mean)
            new_covariance =\
                other.__matmul__(self.covariance.__matmul__(other.transpose()))
        else:
            raise TypeError("Matrix multiplication can only be done with " +\
                "sequence types.")
        return SparseGaussianDistribution(new_mean, new_covariance)
    
    def draw(self, shape=None, random=rand):
        """
        Draws a point from this distribution using `numpy.random`.
        
        Parameters
        ----------
        shape : tuple or int or None
            - if None, returns single random variate as a 1D `numpy.ndarray` of
            length `SparseGaussianDistribution.numparams`
            - if int, \\(n\\), returns \\(n\\) random variates in a 2D
            `numpy.ndarray` of shape
            `(n, SparseGaussianDistribution.numparams)`
            - if tuple of n ints, returns that many random variates in a
            `numpy.ndarray` of shape
            `shape+(SparseGaussianDistribution.numparams,)`
        random : `numpy.random.RandomState`
            the random number generator to use (by default, `numpy.random` is
            used)
        
        Returns
        -------
        random_values : numpy.ndarray
            an array containing the values from this draw. See `shape` argument
            for details on the shape of the return value for different inputs
        """
        if type(shape) is type(None):
            shape = ()
        if type(shape) in int_types:
            shape = (shape,)
        values = random.normal(0, 1, size=shape+(self.numparams,))
        mean_slice = ((np.newaxis,) * len(shape)) + (slice(None),)
        return self.mean[mean_slice] +\
            self.square_root_covariance.__matmul__(values)
    
    @property
    def log_value_constant_part(self):
        """
        The constant part of the log value, i.e. the part of the sum that has
        no dependence on the point at which the distribution is being
        evaluated. It is equal to \\(-\\frac{1}{2}\\left(\
        \\ln{\\left|\\boldsymbol{\\Sigma}\\right|} + N\\ln{(2\\pi)}\\right)\\).
        """
        if not hasattr(self, '_log_value_constant_part'):
            self._log_value_constant_part = (self.log_determinant_covariance +\
            (self.numparams * natural_log_two_pi)) / (-2.)
        return self._log_value_constant_part
    
    def log_value(self, point):
        """
        Evaluates the log of the value of this distribution at the given point.
        
        Parameters
        ----------
        point : numpy.ndarray
            the point, \\(\\boldsymbol{x}\\), in a `numpy.ndarray` of shape
            (self.numparams,)
        
        Returns
        -------
        value : float
            the log of the value of this distribution at the given point. It is
            given by \\(-\\frac{1}{2}\\left[\
            (\\boldsymbol{x}-\\boldsymbol{\\mu})^T\\boldsymbol{\\Sigma}^{-1}\
            (\\boldsymbol{x}-\\boldsymbol{\\mu}) + N\\ln{(2\\pi)} +\
            \\ln{\\left|\\boldsymbol{\\Sigma}\\right|}\\right]\\)
        """
        minus_mean = point - self.mean
        exponent = np.vdot(minus_mean,\
            self.inverse_covariance.__matmul__(minus_mean)) / (-2.)
        return self.log_value_constant_part + exponent
    
    def to_string(self):
        """
        Finds the string representation of this `SparseGaussianDistribution`.
        
        Returns
        -------
        string : str
            `"n-dim Sparse Normal"` where \\(n\\) is the dimension of this
            distribution
        """
        return "{}-dim Sparse Normal".format(self.numparams)
    
    def __eq__(self, other):
        """
        Checks for equality of this distribution with other.
        
        Parameters
        ----------
        other : object
            object to check for equality
        
        Returns
        -------
        result : bool
            True if and only if other is a `SparseGaussianDistribution` with
            the same mean and covariance. Note that this will return False if
            `other` is a
            `distpy.distribution.GaussianDistribution.GaussianDistribution`
            even if it represents the same distribution as this
            `SparseGaussianDistribution`.
        """
        if isinstance(other, SparseGaussianDistribution):
            if self.numparams == other.numparams:
                mean_close =\
                    np.allclose(self.mean, other.mean, rtol=0, atol=1e-9)
                covariance_close = (self.covariance == other.covariance)
                metadata_equal = self.metadata_equal(other)
                return all([mean_close, covariance_close, metadata_equal])
            else:
                return False
        else:
            return False
    
    @property
    def minimum(self):
        """
        The minimum allowable values in this distribution. Since Gaussian
        curves are never zero, there are no minima.
        """
        return ([None] * self.numparams)
    
    @property
    def maximum(self):
        """
        The maximum allowable values in this distribution. Since Gaussian
        curves are never zero, there are no maxima.
        """
        return ([None] * self.numparams)
    
    @property
    def is_discrete(self):
        """
        Boolean (False) describing whether this distribution is discrete.
        """
        return False
    
    def fill_hdf5_group(self, group, mean_link=None, covariance_link=None,\
        save_metadata=True):
        """
        Fills the given hdf5 file group with data from this distribution. The
        fact that this is a GaussianDistribution is saved along with the mean
        and covariance.
        
        Parameters
        ----------
        group: h5py.Group
            hdf5 file group to fill
        mean_link : `distpy.util.h5py_extensions.HDF5Link` or None
            link to extant dataset (or arg or list of args with which to create
            one)
        covariance_link : h5py.Group or None
            group with extant covariance (if it exists)
        save_metadata : bool
            - if True, attempts to save metadata alongside distribution and
            throws error if it fails
            - if False, metadata is ignored in saving process
        """
        group.attrs['class'] = 'SparseGaussianDistribution'
        create_hdf5_dataset(group, 'mean', data=self.mean, link=mean_link)
        if type(covariance_link) is type(None):
            self.covariance.fill_hdf5_group(group.create_group('covariance'))
        elif isinstance(covariance_link, basestring):
            group['covariance'] = h5py.SoftLink(covariance_link)
        elif isinstance(covariance_link, h5py.Group):
            group['covariance'] = covariance_link
        if save_metadata:
            self.save_metadata(group)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a `SparseGaussianDistribution` from the given hdf5 file group.
        
        Parameters
        ----------
        group: h5py.Group
            hdf5 file group from which to load a distribution
        
        Returns
        -------
        loaded : `SparseGaussianDistribution`
            distribution loaded from the given hdf5 file group
        """
        try:
            assert group.attrs['class'] == 'SparseGaussianDistribution'
        except:
            raise TypeError("The given hdf5 file doesn't seem to contain a " +\
                "SparseGaussianDistribution.")
        metadata = Distribution.load_metadata(group)
        mean = get_hdf5_value(group['mean'])
        covariance = SparseSquareBlockDiagonalMatrix.load_from_hdf5_group(\
            group['covariance'])
        return SparseGaussianDistribution(mean, covariance, metadata=metadata)
    
    @property
    def gradient_computable(self):
        """
        Boolean (True) describing whether this distribution's gradient can be
        taken.
        """
        return True
    
    def gradient_of_log_value(self, point):
        """
        Computes the derivatives of log_value(point) with respect to the
        parameters.
        
        Parameters
        ----------
        point : numpy.ndarray
            1D array of length given by `SparseGaussianDistribution.numparams`
            containing parameter values at which to evaluate distribution,
            \\(\\boldsymbol{x}\\)
        
        Returns
        -------
        gradient : numpy.ndarray
            1D vector of derivatives of log value, given by
            \\(-\\boldsymbol{\\Sigma}^{-1}\
            (\\boldsymbol{x}-\\boldsymbol{\\mu}),\\) where \\(x\\) is the value
            of `point`
        """
        return self.inverse_covariance.__matmul__(self.mean - point)
    
    @property
    def hessian_computable(self):
        """
        Boolean (True) describing whether this distribution's hessian can be
        taken.
        """
        return True
    
    def hessian_of_log_value(self, point):
        """
        Computes the second derivatives of log_value(point) with respect to the
        parameters.
        
        Parameters
        ----------
        point : numpy.ndarray
            1D array of length given by `SparseGaussianDistribution.numparams`
            containing parameter values at which to evaluate distribution
        
        Returns
        -------
        hessian : numpy.ndarray
            2D square matrix of second derivatives of log value, given by
            \\(-\\boldsymbol{\\Sigma}^{-1}\\) regardless of the value of point
        """
        return -self.inverse_covariance.dense()
    
    def copy(self):
        """
        Returns a deep copy of this `SparseGaussianDistribution`. This function
        ignores metadata.
        
        Returns
        -------
        copied : `SparseGaussianDistribution`
            a deep copy of this distribution, excluding metadata
        """
        return SparseGaussianDistribution(self.mean.copy(),\
            self.covariance.copy())

