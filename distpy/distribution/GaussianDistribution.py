"""
Module containing class representing a Gaussian distribution. Its PDF is
represented by: $$f(\\boldsymbol{x})=|2\\pi\\boldsymbol{\\Sigma}|^{-1/2}\\ \
\\exp{\\left[-\\frac{1}{2}(\\boldsymbol{x}-\\boldsymbol{\\mu})^T\
\\boldsymbol{\\Sigma}^{-1}(\\boldsymbol{x}-\\boldsymbol{\\mu})\\right]}$$

**File**: $DISTPY/distpy/distribution/GaussianDistribution.py  
**Author**: Keith Tauscher  
**Date**: 31 May 2021
"""

from __future__ import division
import numpy as np
import numpy.random as rand
import numpy.linalg as npla
import scipy.linalg as scila
from scipy.special import erfinv
from ..util import numerical_types, int_types, sequence_types,\
    create_hdf5_dataset, get_hdf5_value
from .Distribution import Distribution

natural_log_two_pi = np.log(2 * np.pi)

class GaussianDistribution(Distribution):
    """
    Class representing a Gaussian distribution. Its PDF is represented by:
    $$f(\\boldsymbol{x})=|2\\pi\\boldsymbol{\\Sigma}|^{-1/2}\\ \
    \\exp{\\left[-\\frac{1}{2}(\\boldsymbol{x}-\\boldsymbol{\\mu})^T\
    \\boldsymbol{\\Sigma}^{-1}(\\boldsymbol{x}-\\boldsymbol{\\mu})\\right]}$$
    """
    def __init__(self, mean, covariance, metadata=None):
        """
        Initializes a new `GaussianDistribution` with the given parameter
        values.
        
        Parameters
        ----------
        mean : float or `numpy.ndarray`
            - if this is a univariate Gaussian, `mean` is a real number giving
            peak of distribution
            - if this is a multivariate Gaussian, `mean` is a 1D array of real
            numbers giving peak of distribution
        covariance : float or `numpy.ndarray`
            - if this is a univariate Gaussian, `covariance` is a real,
            positive number giving size of distribution
            - if this is a multivariate Gaussian, `covariance` is a square 2D
            array giving covariance matrix of the distribution. Each dimension
            should have the same length as `mean`
        metadata : number or str or dict or `distpy.util.Savable.Savable`
            data to store alongside this distribution.
        """
        self.internal_mean = mean
        self.covariance = covariance
        self.metadata = metadata
    
    @staticmethod
    def combine(*distributions):
        """
        Combines many `GaussianDistribution` objects into one by concatenating
        their means and covariance matrices.
        
        Parameters
        ----------
        distributions : sequence
            a sequence of `GaussianDistribution` objects to combine
        
        Returns
        -------
        combined : `GaussianDistribution`
            if the distributions in `distributions` have means
            \\(\\boldsymbol{\\mu}_1,\\boldsymbol{\\mu}_2,\\ldots,\
            \\boldsymbol{\\mu}_N\\) and covariances
            \\(\\boldsymbol{\\Sigma}_1,\\boldsymbol{\\Sigma}_2,\\ldots,\
            \\boldsymbol{\\Sigma}_N\\), then `combined` has mean
            \\(\\begin{bmatrix} \\boldsymbol{\\mu}_1 \\\\\
            \\boldsymbol{\\mu}_2 \\\\ \\vdots \\\\ \\boldsymbol{\\mu}_N\
            \\end{bmatrix}\\) and covariance \\(\\begin{bmatrix}\
            \\boldsymbol{\\Sigma}_1 & \\boldsymbol{0} & \\cdots &\
            \\boldsymbol{0} \\\\ \\boldsymbol{0} & \\boldsymbol{\\Sigma}_2 &\
            \\cdots & \\boldsymbol{0} \\\\ \\vdots & \\vdots & \\ddots &\
            \\vdots \\\\ \\boldsymbol{0} & \\boldsymbol{0} & \\cdots &\
            \\boldsymbol{\\Sigma}_N \\end{bmatrix}\\)
        """
        if all([isinstance(distribution, GaussianDistribution)\
            for distribution in distributions]):
            new_mean = np.concatenate([distribution.internal_mean.A[0]\
                for distribution in distributions])
            new_covariance = scila.block_diag(*[distribution.covariance.A\
                for distribution in distributions])
            return GaussianDistribution(new_mean, new_covariance)
        else:
            raise TypeError("At least one of the distributions given to " +\
                "the GaussianDistribution class' combine function was not " +\
                "a GaussianDistribution.")
    
    @property
    def internal_mean(self):
        """
        The mean of this `GaussianDistribution` in `numpy.matrix` form.
        """
        if not hasattr(self, '_internal_mean'):
            raise AttributeError("internal_mean was referenced before it " +\
                "was set.")
        return self._internal_mean
    
    @internal_mean.setter
    def internal_mean(self, value):
        """
        Setter for `GaussianDistribution.internal_mean`.
        
        Parameters
        ----------
        value : float or `numpy.ndarray`
            - if this distribution is univariate, `value` is a single number
            - otherwise, `value` is a 1D numpy.ndarray of length
            `GaussianDistribution.numparams`
        """
        if type(value) in numerical_types:
            value = [value]
        if type(value) in sequence_types:
            value = np.array(value)
            if value.ndim != 1:
                raise ValueError("The mean of a GaussianDistribution was " +\
                    "not 1 dimensional.")
            elif value.size == 0:
                raise ValueError("The mean of a GaussianDistribution was " +\
                    "set to something like an empty array.")
            else:
                self._internal_mean = np.matrix(value)
        else:
            raise ValueError("The mean of a GaussianDistribution is not of " +\
                "a recognizable type.")
    
    @property
    def covariance(self):
        """
        The covariance matrix of this `GaussianDistribution` in `numpy.matrix`
        form.
        """
        if not hasattr(self, '_covariance'):
            raise AttributeError("covariance was referenced before it was " +\
                "set.")
        return self._covariance
    
    @covariance.setter
    def covariance(self, value):
        """
        Setter for the `GaussianDistribution.covariance`.
        
        Parameters
        ----------
        value : float or numpy.ndarray
            - if this distribution is univariate, then `value` can be a single
            number representing the variance
            - otherwise, this should be a square positive definite matrix of
            rank numparams or a 1D array of variances (in which case the
            variates are assumed independent)
        """
        if type(value) in numerical_types:
            if self.numparams == 1:
                self._covariance = np.matrix([[value]])
            else:
                raise TypeError("covariance was set to a number even " +\
                    "though this Gaussian is multi-dimensional.")
        elif type(value) in sequence_types:
            value = np.array(value)
            if np.any(np.isnan(value)):
                raise ValueError(("For some reason, there are nan's in the " +\
                    "covariance matrix given to a GaussianDistribution, " +\
                    "which was:\n{}.").format(value))
            elif value.shape == (self.numparams,):
                self._covariance = np.matrix(np.diag(value))
            elif value.shape == ((self.numparams,) * 2):
                self._covariance = np.matrix((value + value.T) / 2)
            else:
                raise ValueError("The covariance given to a " +\
                    "GaussianDistribution was not castable to an array of " +\
                    "the correct shape. It should be a square shape with " +\
                    "the same side length as length of mean.")
        else:
            raise ValueError("The mean of a GaussianDistribution is " +\
                "array-like but its covariance isn't matrix like.")
        self.square_root_covariance
    
    @property
    def mean(self):
        """
        The mean of this `GaussianDistribution`, \\(\\boldsymbol{\\mu}\\),
        which is an array if this distribution is multivariate and a scalar if
        it is univariate.
        """
        if not hasattr(self, '_mean'):
            if self.numparams == 1:
                self._mean = self.internal_mean.A[0,0]
            else:
                self._mean = self.internal_mean.A[0]
        return self._mean
    
    @property
    def variance(self):
        """
        The (co)variance of this `GaussianDistribution`,
        \\(\\boldsymbol{\\Sigma}\\).
        """
        if not hasattr(self, '_variance'):
            if self.numparams == 1:
                self._variance = self.covariance.A[0,0]
            else:
                self._variance = self.covariance.A
        return self._variance
    
    @property
    def log_determinant_covariance(self):
        """
        The natural logarithm of the determinant of the covariance matrix,
        given by \\(\\ln{|\\boldsymbol{\\Sigma}|}\\).
        """
        if not hasattr(self, '_log_determinant_covariance'):
            self._log_determinant_covariance = npla.slogdet(self.covariance)[1]
        return self._log_determinant_covariance
    
    @property
    def inverse_covariance(self):
        """
        The inverse of the covariance matrix, given by
        \\(\\boldsymbol{\\Sigma}^{-1}\\).
        """
        if not hasattr(self, '_inverse_covariance'):
            if self.covariance_diagonal:
                self._inverse_covariance =\
                    np.matrix(np.diag(1 / np.diag(self.covariance.A)))
            else:
                self._inverse_covariance = npla.inv(self.covariance)
        return self._inverse_covariance
    
    @property
    def numparams(self):
        """
        The number of parameters of this `GaussianDistribution`.
        """
        if not hasattr(self, '_numparams'):
            self._numparams = len(self.internal_mean.A[0])
        return self._numparams
    
    def __add__(self, other):
        """
        Adds the Gaussian random variate described by this distribution
        to the given object.
        
        Parameters
        ----------
        other : float or numpy.ndarray or `GaussianDistribution`
            - if other is a constant, the returned Gaussian is the same as this
            one with other added to the mean and the same covariance
            - if other is a 1D `numpy.ndarray`, it must be of the same length
            as the dimension of this `GaussianDistribution`. In this case, the
            returned `GaussianDistribution` is the distribution of the sum of
            this Gaussian variate with other
            - if other is a `GaussianDistribution`, it must have the same
            number of parameters as this one
        
        Returns
        -------
        distribution : `GaussianDistribution`
            distribution of the sum of this Gaussian variate and `other`
        """
        if isinstance(other, GaussianDistribution):
            if self.numparams == other.numparams:
                new_mean = self.internal_mean.A[0] + other.internal_mean.A[0]
                new_covariance = self.covariance.A + other.covariance.A
            else:
                raise ValueError("Cannot add two GaussianDistribution " +\
                    "objects with different numbers of parameters.")
        elif type(other) in [list, tuple, np.ndarray]:
            other = np.array(other)
            if other.ndim == 1:
                if len(other) == self.numparams:
                    new_mean = self.internal_mean.A[0] + other
                    new_covariance = self.covariance.A
                else:
                    raise ValueError("Cannot multiply Gaussian distributed " +\
                        "random vector by a vector of different size.")
            else:
                raise ValueError("Cannot multiply Gaussian distributed " +\
                    "random vector by a tensor with more than 1 index.")
        else:
            # assume other is a constant
            new_mean = self.internal_mean.A[0] + other
            new_covariance = self.covariance.A
        return GaussianDistribution(new_mean, new_covariance)
    
    def __radd__(self, other):
        """
        Adds the Gaussian random variate described by this distribution
        to the given object.
        
        Parameters
        ----------
        other : float or numpy.ndarray or `GaussianDistribution`
            - if other is a constant, the returned Gaussian is the same as this
            one with other added to the mean and the same covariance
            - if other is a 1D `numpy.ndarray`, it must be of the same length
            as the dimension of this `GaussianDistribution`. In this case, the
            returned `GaussianDistribution` is the distribution of the sum of
            this Gaussian variate with other
            - if other is a `GaussianDistribution`, it must have the same
            number of parameters as this one
        
        Returns
        -------
        distribution : `GaussianDistribution`
            distribution of the sum of this Gaussian variate and `other`
        """
        return self.__add__(other)
    
    def __sub__(self, other):
        """
        Subtracts the given object from the Gaussian random variate described
        by this distribution.
        
        Parameters
        ----------
        other : float or numpy.ndarray or `GaussianDistribution`
            - if other is a constant, the returned Gaussian is the same as this
            one with other subtracted from the mean and the same covariance
            - if other is a 1D `numpy.ndarray`, it must be of the same length
            as the dimension of this `GaussianDistribution`. In this case, the
            returned `GaussianDistribution` is the distribution of the
            difference of this Gaussian variate with `other`
            - if other is a `GaussianDistribution`, it must have the same
            number of parameters as this one
        
        Returns
        -------
        distribution : `GaussianDistribution`
            distribution of the difference of this Gaussian variate and `other`
        """
        return self.__add__(-other)
    
    def __rsub__(self, other):
        """
        Subtracts the Gaussian random variate described by this distribution
        from `other`.
        
        Parameters
        ----------
        other : float or numpy.ndarray or `GaussianDistribution`
            - if other is a constant, the returned Gaussian is the same as this
            one with mean replaces with other-mean and the same covariance
            - if other is a 1D `numpy.ndarray`, it must be of the same length
            as the dimension of this `GaussianDistribution`. In this case, the
            returned `GaussianDistribution` is the distribution of the
            difference of this Gaussian variate with `other`
            - if other is a `GaussianDistribution`, it must have the same
            number of parameters as this one
        
        Returns
        -------
        distribution : `GaussianDistribution`
            distribution of the difference of this Gaussian variate and `other`
        """
        return self.__sub__(other).__neg__()
    
    def __neg__(self):
        """
        Finds the distribution of the negated gaussian variate.
        
        Returns
        -------
        distribution : `GaussianDistribution`
            distribution with the same covariance but a negated mean
        """
        return GaussianDistribution(-self.internal_mean.A[0],\
            self.covariance.A)
    
    def __mul__(self, other):
        """
        Multiplies the Gaussian random variate described by this distribution
        by the given constant.
        
        Parameters
        ----------
        other : float
            any real number
        
        Returns
        -------
        distribution : `GaussianDistribution`
            distribution of the product of the random variable with this
            distribution and the constant `other`
        """
        new_mean = self.internal_mean.A[0] * other
        new_covariance = self.covariance.A * (other ** 2)
        return GaussianDistribution(new_mean, new_covariance)
    
    def __rmul__(self, other):
        """
        Multiplies the Gaussian random variate described by this distribution
        by the given constant.
        
        Parameters
        ----------
        other : float
            any real number
        
        Returns
        -------
        distribution : `GaussianDistribution`
            distribution of the product of the random variable with this
            distribution and the constant `other`
        """
        return self.__mul__(other)
    
    def __div__(self, other):
        """
        Divides the Gaussian random variate described by this distribution
        by the given constant.
        
        Parameters
        ----------
        other : float
            any real number
        
        Returns
        -------
        distribution : `GaussianDistribution`
            distribution of the quotient of the random variable with this
            distribution and the constant `other`
        """
        return self.__mul__(1 / other)
    
    @property
    def covariance_diagonal(self):
        """
        A boolean describing whether the covariance matrix is exactly diagonal
        or not.
        """
        if not hasattr(self, '_covariance_diagonal'):
            self._covariance_diagonal = np.all(\
                self.covariance.A == np.diag(np.diag(self.covariance.A)))
        return self._covariance_diagonal
    
    def _make_square_root_and_inverse_square_root_covariance(self):
        """
        Computes the square root and inverse square root of the covariance
        matrix and stores them in internal properties, allowing
        `GaussianDistribution.square_root_covariance` and
        `GaussianDistribution.inverse_square_root_covariance` properties to be
        referenced.
        """
        if self.covariance_diagonal:
            self._square_root_covariance =\
                np.diag(np.sqrt(np.diag(self.covariance.A)))
            self._inverse_square_root_covariance =\
                np.diag(1 / np.sqrt(np.diag(self.covariance.A)))
        else:
            (eigenvalues, eigenvectors) = npla.eigh(self.covariance.A)
            if np.any(eigenvalues <= 0):
                raise ValueError(("Something went wrong, causing the " +\
                    "square root of the covariance matrix of this " +\
                    "GaussianDistribution to have at least one complex " +\
                    "element. The eigenvalues of the covariance matrix " +\
                    "are {!s}.").format(eigenvalues))
            eigenvalues = np.sqrt(eigenvalues)
            self._square_root_covariance =\
                np.dot(eigenvectors * eigenvalues[None,:], eigenvectors.T)
            self._inverse_square_root_covariance =\
                np.dot(eigenvectors / eigenvalues[None,:], eigenvectors.T)
    
    @property
    def square_root_covariance(self):
        """
        The square root of the covariance matrix, given by
        \\(\\boldsymbol{\\Sigma}^{1/2}\\).
        """
        if not hasattr(self, '_square_root_covariance'):
            self._make_square_root_and_inverse_square_root_covariance()
        return self._square_root_covariance
    
    @property
    def inverse_square_root_covariance(self):
        """
        The inverse of the square root of the covariance matrix, given by
        \\(\\boldsymbol{\\Sigma}^{-1/2}\\).
        """
        if not hasattr(self, '_inverse_square_root_covariance'):
            self._make_square_root_and_inverse_square_root_covariance()
        return self._inverse_square_root_covariance
    
    def weight(self, array, axis=0):
        """
        Weights the given array by the inverse square root of the covariance
        matrix of this distribution.
        
        Parameters
        ----------
        array : numpy.ndarray
            the array to weight, can be any number of dimensions as long as the
            specified one has length `GaussianDistribution.numparams`
        axis : int
           index of the axis corresponding to the parameters
        
        Returns
        -------
        weighted : numpy.ndarray
            `numpy.ndarray` of same shape as `array` corresponding to
            \\(\\boldsymbol{\\Sigma}^{-1/2}\\boldsymbol{A}\\), where
            \\(\\boldsymbol{A}\\) is `array` shaped so that the matrix
            multiplication makes sense.
        """
        axis = axis % array.ndim
        if self.covariance_diagonal:
            error_slice = ((None,) * axis) + (slice(None),) +\
                ((None,) * (array.ndim - axis - 1))
            return array * self.inverse_square_root_covariance[error_slice]
        elif array.ndim == 1:
            return np.dot(self.inverse_square_root_covariance, array)
        elif array.ndim == 2:
            if axis == 0:
                return np.dot(self.inverse_square_root_covariance, array)
            else:
                return np.dot(array, self.inverse_square_root_covariance)
        else:
            before_shape = array.shape[:axis]
            after_shape = array.shape[(axis+1):]
            if axis != 0:
                weighted_array = np.rollaxis(array, axis, start=0)
            weighted_array = np.reshape(weighted_array, (self.numparams, -1))
            weighted_array =\
                np.dot(self.inverse_square_root_covariance, weighted_array)
            weighted_array = np.reshape(weighted_array,\
                (self.numparams,) + before_shape + after_shape)
            if axis != 0:
                weighted_array = np.rollaxis(weighted_array, 0, start=axis+1)
            return weighted_array
    
    def __matmul__(self, other):
        """
        Finds and returns the distribution of the matrix product of other with
        the random variable this distribution describes.
        
        Parameters
        ----------
        other : numpy.ndarray
            - if other is a 1D numpy.ndarray, it must be of the same length as
            the dimension of this `GaussianDistribution`. In this case, the
            returned `GaussianDistribution` is the distribution of the dot
            product of this Gaussian variate with `other`
            - if other is a 2D numpy.ndarray, it must have shape
            `(newparams, self.numparams)` where `newparams<=self.numparams`.
            The returned `GaussianDistribution` is the distribution of `other`
            (matrix) multiplied with this Gaussian variate
        
        Returns
        -------
        distribution : `GaussianDistribution`
            distribution of matrix multiplication of `other` and the random
            variate this distribution represents
        """
        if type(other) in [list, tuple, np.ndarray]:
            other = np.array(other)
            if other.ndim == 1:
                if len(other) == self.numparams:
                    new_mean = np.dot(self.internal_mean.A[0], other)
                    new_covariance =\
                        np.dot(np.dot(self.covariance.A, other), other)
                else:
                    raise ValueError("Cannot multiply Gaussian distributed " +\
                        "random vector by a vector of different size.")
            elif other.ndim == 2:
                if other.shape[1] == self.numparams:
                    if other.shape[0] <= self.numparams:
                        # other is a matrix with self.numparams columns
                        new_mean = np.dot(other, self.internal_mean.A[0])
                        new_covariance =\
                            np.dot(other, np.dot(self.covariance.A, other.T))
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
        else:
            raise TypeError("Matrix multiplication can only be done with " +\
                "sequence types.")
        return GaussianDistribution(new_mean, new_covariance)
    
    @staticmethod
    def kullback_leibler_divergence(first, second):
        """
        Computes the Kullback-Leibler divergence between two distributions
        represented by `GaussianDistribution` objects.
        
        Parameters
        ----------
        first : numpy.ndarray or `GaussianDistribution`
            distribution (or just covariance matrix) to find divergence from
        second : numpy.ndarray or `GaussianDistribution`
            distribution (or just covariance matrix) to find divergence to
        
        Returns
        -------
        divergence : float
            the Kullback-Leibler divergence from `first` to `second`. If
            `first` and `second` are covariance matrices, then the term
            corresponding to the mean difference is omitted.
        """
        if isinstance(first, GaussianDistribution) and\
            isinstance(second, GaussianDistribution):
            if first.numparams == second.numparams:
                dimension = first.numparams
                delta = first.internal_mean.A[0] - second.internal_mean.A[0]
                sigma_Q_inverse = npla.inv(second.covariance.A)
                sigma_P_times_sigma_Q_inverse =\
                    np.dot(first.covariance.A, sigma_Q_inverse)
                return ((np.sum(np.diag(sigma_P_times_sigma_Q_inverse)) -\
                    dimension -\
                    npla.slogdet(sigma_P_times_sigma_Q_inverse)[1] +\
                    np.dot(delta, np.dot(sigma_Q_inverse, delta))) / 2)
            else:
                raise ValueError("The two given distributions do not have " +\
                    "the same numbers of parameters.")
        elif isinstance(first, np.ndarray) and isinstance(second, np.ndarray):
            if first.shape == second.shape:
                if (first.ndim == 2) and (first.shape[0] == first.shape[1]):
                    mean = np.zeros(first.shape[0])
                    first_distribution = GaussianDistribution(mean, first)
                    second_distribution = GaussianDistribution(mean, second)
                    return GaussianDistribution.kullback_leibler_divergence(\
                        first_distribution, second_distribution)
                else:
                    raise ValueError("The covariance matrices given to the " +\
                        "GaussianDistribution class' " +\
                        "kullback_leibler_divergence function were not 2D " +\
                        "square.")
            else:
                raise ValueError("The shapes of the two covariance " +\
                    "matrices given to the GaussianDistribution class' " +\
                    "kullback_leibler_divergence function were not of the " +\
                    "same shape.")
        else:
            raise TypeError("At least one of the distributions given to " +\
                "the kullback_leibler_divergence static method of the " +\
                "GaussianDistribution class was not a GaussianDistribution " +\
                "object.")
    
    def draw(self, shape=None, random=rand):
        """
        Draws point(s) from this `GaussianDistribution`.
        
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
        if (self.numparams == 1):
            loc = self.internal_mean.A[0,0]
            scale = np.sqrt(self.covariance.A[0,0])
            return random.normal(loc=loc, scale=scale, size=shape)
        elif type(shape) is type(None):
            if self.covariance_diagonal:
                return self.internal_mean.A[0] +\
                    (np.diag(self.square_root_covariance) *\
                    random.normal(0, 1, size=self.numparams))
            else:
                return self.internal_mean.A[0] +\
                    np.dot(self.square_root_covariance,\
                    random.normal(0, 1, size=self.numparams))
        else:
            if type(shape) in int_types:
                shape = (shape,)
            if self.covariance_diagonal:
                random_vector =\
                    random.normal(0, 1, size=shape+(self.numparams,))
                return self.internal_mean.A +\
                    (random_vector * np.diag(self.square_root_covariance))
            else:
                random_vector =\
                    random.normal(0, 1, size=shape+(1,self.numparams))
                return self.internal_mean.A + np.sum(random_vector *\
                    self.square_root_covariance, axis=-1)
    
    @property
    def log_value_constant_part(self):
        """
        The constant part of the log value, i.e. the part of the sum that has
        no dependence on the point at which the distribution is being
        evaluated. It is given by
        \\(-\\frac{1}{2}\\ln{|\\boldsymbol{\\Sigma}|}-\
        \\frac{N}{2}\\ln{2\\pi}\\).
        """
        if not hasattr(self, '_log_value_constant_part'):
            self._log_value_constant_part = (self.log_determinant_covariance +\
            (self.numparams * natural_log_two_pi)) / (-2.)
        return self._log_value_constant_part
    
    def log_value(self, point):
        """
        Computes the logarithm of the value of this `GaussianDistribution` at
        the given point.
        
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
        if type(point) in numerical_types:
            minus_mean = np.matrix([point]) - self.internal_mean
        elif type(point) in sequence_types:
            minus_mean = np.matrix(point) - self.internal_mean
        else:
            raise ValueError("The type of point provided to a " +\
                "GaussianDistribution was not of a numerical type " +\
                "(should be if distribution is univariate) or of a " +\
                "list type (should be if distribution is multivariate).")
        if self.covariance_diagonal:
            exponent = np.sum((minus_mean.A[0] ** 2) /\
                np.diag(self.covariance.A)) / (-2.)
        else:
            exponent = np.float64(\
                minus_mean * self.inverse_covariance * minus_mean.T) / (-2.)
        return self.log_value_constant_part + exponent
    
    def to_string(self):
        """
        Finds and returns a string version of this `GaussianDistribution` of
        the form `"Normal(mean=mu, variance=sigma2)"` or `"d-dim Normal"`.
        """
        if self.numparams == 1:
            return "Normal(mean={0:.3g},variance={1:.3g})".format(\
                self.internal_mean.A[0,0], self.covariance.A[0,0])
        else:
            return "{}-dim Normal".format(len(self.internal_mean.A[0]))
    
    def marginalize(self, key):
        """
        Marginalizes this Gaussian over all of the parameters not described by
        given key.
        
        Parameters
        ----------
        key : int or numpy.ndarray or slice
            key representing index (indices) to keep
        
        Returns
        -------
        marginalized : `GaussianDistribution`
            distribution of the desired parameters marginalized over other
            parameters
        """
        new_mean = self.internal_mean.A[0][key]
        new_covariance = self.covariance.A[:,key][key]
        return GaussianDistribution(new_mean, new_covariance)
    
    def __getitem__(self, key):
        """
        Marginalizes this Gaussian over all of the parameters not described by
        given key.
        
        Parameters
        ----------
        key : int or numpy.ndarray or slice
            key representing index (indices) to keep
        
        Returns
        -------
        marginalized : `GaussianDistribution`
            distribution of the desired parameters marginalized over other
            parameters
        """
        return self.marginalize(key)
    
    def conditionalize(self, known_indices, values):
        """
        Conditionalized this Gaussian over all of the parameters not described
        by given `known_indices`.
        
        Parameters
        ----------
        known_indices : int or numpy.ndarray or slice
            key representing index (indices) to keep
        values : numpy.ndarray
            values of variables corresponding to `known_indices`
        
        Returns
        -------
        conditionalized : `GaussianDistribution`
            distribution when the parameters corresponding to `known_indices`
            are fixed to `values`
        """
        if isinstance(known_indices, slice):
            known_indices = np.arange(*known_indices.indices(self.numparams))
        elif type(known_indices) in int_types:
            known_indices = np.array([known_indices])
        elif type(known_indices) in sequence_types:
            known_indices = np.array(known_indices)
        remaining_indices = np.array([index\
            for index in np.arange(self.numparams)\
            if index not in known_indices])
        new_covariance = npla.inv(\
            self.inverse_covariance.A[:,remaining_indices][remaining_indices])
        known_mean_displacement =\
            values - self.internal_mean.A[0][known_indices]
        new_mean =\
            self.internal_mean.A[0][remaining_indices] -\
            np.dot(new_covariance, np.dot(\
            self.inverse_covariance.A[:,known_indices][remaining_indices],\
            known_mean_displacement))
        return GaussianDistribution(new_mean, new_covariance)
    
    def __eq__(self, other):
        """
        Checks for equality of this `GaussianDistribution` with `other`.
        
        Parameters
        ----------
        other : object
            object to check for equality
        
        Returns
        -------
        result : bool
            True if and only if `other` is a `GaussianDistribution` with the
            same mean and variance
        """
        if isinstance(other, GaussianDistribution):
            if self.numparams == other.numparams:
                mean_close = np.allclose(self.internal_mean.A,\
                    other.internal_mean.A, rtol=0, atol=1e-9)
                covariance_close = np.allclose(self.covariance.A,\
                    other.covariance.A, rtol=1e-12, atol=0)
                metadata_equal = self.metadata_equal(other)
                return all([mean_close, covariance_close, metadata_equal])
            else:
                return False
        else:
            return False
    
    def inverse_cdf(self, cdf):
        """
        Computes the inverse of the cumulative distribution function (cdf) of
        this `GaussianDistribution`. Only valid when
        `GaussianDistribution.numparams` is 1.
        
        Parameters
        ----------
        cdf : float
            probability value between 0 and 1
        
        Returns
        -------
        point : float
            value which yields `cdf` when it the CDF is evaluated at it
        """
        return (self.internal_mean.A[0,0] +\
            (np.sqrt(2 * self.covariance.A[0,0]) * erfinv((2 * cdf) - 1)))
    
    @property
    def minimum(self):
        """
        The minimum allowable value(s) in this distribution.
        """
        return None if (self.numparams == 1) else ([None] * self.numparams)
    
    @property
    def maximum(self):
        """
        The maximum allowable value(s) in this distribution.
        """
        return None if (self.numparams == 1) else ([None] * self.numparams)
    
    @property
    def is_discrete(self):
        """
        Boolean describing whether this distribution is discrete (True) or
        continuous (False).
        """
        return False
    
    def fill_hdf5_group(self, group, mean_link=None, covariance_link=None,\
        save_metadata=True):
        """
        Fills the given hdf5 file group with data about this
        `GaussianDistribution` so that it can be loaded later.
        
        Parameters
        ----------
        group : h5py.Group
            hdf5 file group to fill
        mean_link : str or h5py.Dataset or None
            link to mean vector in hdf5 file, if it exists
        covariance_link : str or h5py.Dataset or None
            link to mean vector in hdf5 file, if it exists
        save_metadata : bool
            - if True, attempts to save metadata alongside distribution and
            throws error if it fails
            - if False, metadata is ignored in saving process
        """
        group.attrs['class'] = 'GaussianDistribution'
        create_hdf5_dataset(group, 'mean', data=self.internal_mean.A[0],\
            link=mean_link)
        create_hdf5_dataset(group, 'covariance', data=self.covariance.A,\
            link=covariance_link)
        if save_metadata:
            self.save_metadata(group)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a `GaussianDistribution` from the given hdf5 file group.
        
        Parameters
        ----------
        group : h5py.Group
            the same hdf5 file group which fill_hdf5_group was called on when
            this Distribution was saved
        
        Returns
        -------
        distribution : `GaussianDistribution`
            distribution created from the information in the given group
        """
        try:
            assert group.attrs['class'] == 'GaussianDistribution'
        except:
            raise TypeError("The given hdf5 file doesn't seem to contain a " +\
                "GaussianDistribution.")
        metadata = Distribution.load_metadata(group)
        mean = get_hdf5_value(group['mean'])
        covariance = get_hdf5_value(group['covariance'])
        return GaussianDistribution(mean, covariance, metadata=metadata)
    
    @property
    def gradient_computable(self):
        """
        Boolean describing whether the gradient of the given distribution has
        been implemented. If True,
        `GaussianDistribution.gradient_of_log_value` method can be called
        safely.
        """
        return True
    
    def gradient_of_log_value(self, point):
        """
        Computes the gradient (derivative) of the logarithm of the value of
        this `GaussianDistribution` at the given point.
        
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
        if type(point) in numerical_types:
            mean_minus = self.internal_mean - np.matrix([point])
        elif type(point) in sequence_types:
            mean_minus = self.internal_mean - np.matrix(point)
        else:
            raise ValueError("The type of point provided to a " +\
                "GaussianDistribution was not of a numerical type (should " +\
                "be if distribution is univariate) or of a list type " +\
                "(should be if distribution is multivariate).")
        if self.numparams == 1:
            return (mean_minus * self.inverse_covariance).A[0,0]
        elif self.covariance_diagonal:
            return mean_minus.A[0] / np.diag(self.covariance.A)
        else:
            return (mean_minus * self.inverse_covariance).A[0,:]
    
    @property
    def hessian_computable(self):
        """
        Boolean describing whether the hessian of the given distribution has
        been implemented. If True,
        `GaussianDistribution.hessian_of_log_value` method can be called
        safely.
        """
        return True
    
    def hessian_of_log_value(self, point):
        """
        Computes the hessian (second derivative) of the logarithm of the value
        of this `GaussianDistribution` at the given point.
        
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
        if self.numparams == 1:
            return -self.inverse_covariance.A[0,0]
        else:
            return -self.inverse_covariance.A
    
    def copy(self):
        """
        Copies this distribution.
        
        Returns
        -------
        copied : `GaussianDistribution`
            a deep copy of this distribution, ignoring metadata.
        """
        return GaussianDistribution(self.internal_mean.A[0].copy(),\
            self.covariance.A.copy())
