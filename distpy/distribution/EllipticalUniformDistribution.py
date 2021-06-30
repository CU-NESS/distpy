"""
Module containing class representing a distribution that is uniform over an
ellipsoid in an arbitrary number of dimensions. Its PDF is represented by:
$$f(\\boldsymbol{x})=\\begin{cases}\\Gamma\\left(\\frac{N}{2}+1\\right)\
\\pi^{-N/2}\\Vert\\boldsymbol{\\Sigma}\\Vert^{-1/2}(N+2)^{-N/2} &\
(\\boldsymbol{x}-\\boldsymbol{\\mu})^T\\boldsymbol{\\Sigma}^{-1}\
(\\boldsymbol{x}-\\boldsymbol{\\mu}) \\le N+2 \\\\ 0 & \\text{otherwise}\
\\end{cases},$$ where \\(N=\\text{dim}(\\boldsymbol{x})\\) and \\(\\Gamma(x)\\)
is the Gamma function.

**File**: $DISTPY/distpy/distribution/EllipticalUniformDistribution.py  
**Author**: Keith Tauscher  
**Date**: 31 May 2021
"""
from __future__ import division
import numpy as np
import numpy.random as rand
import numpy.linalg as lalg
import scipy.linalg as slalg
from scipy.special import gammaln as log_gamma
from ..util import int_types, sequence_types, numerical_types,\
    create_hdf5_dataset, get_hdf5_value
from .Distribution import Distribution

class EllipticalUniformDistribution(Distribution):
    """
    Class representing a distribution that is uniform over an ellipsoid in an
    arbitrary number of dimensions. Its PDF is represented by:
    $$f(\\boldsymbol{x})=\\begin{cases}\\Gamma\\left(\\frac{N}{2}+1\\right)\
    \\pi^{-N/2}\\Vert\\boldsymbol{\\Sigma}\\Vert^{-1/2}(N+2)^{-N/2} &\
    (\\boldsymbol{x}-\\boldsymbol{\\mu})^T\\boldsymbol{\\Sigma}^{-1}\
    (\\boldsymbol{x}-\\boldsymbol{\\mu}) \\le N+2 \\\\ 0 & \\text{otherwise}\
    \\end{cases},$$ where \\(N=\\text{dim}(\\boldsymbol{x})\\) and
    \\(\\Gamma(x)\\) is the Gamma function.
    """
    def __init__(self, mean, covariance, metadata=None):
        """
        Initializes a new `EllipticalUniformDistribution` with the given
        parameter values.
        
        Parameters
        ----------
        mean : `numpy.ndarray`
            1D vector, \\(\\boldsymbol{\\mu}\\), defining location of ellipsoid
            by its center
        covariance : `numpy.ndarray`
            2D array, \\(\\boldsymbol{\\Sigma}\\), defining shape of ellipsoid
            by its covariance values
        metadata : number or str or dict or `distpy.util.Savable.Savable`
            data to store alongside this distribution.
        """
        self.mean = mean
        self.covariance = covariance
        self.square_root_covariance
        self.metadata = metadata
    
    @property
    def mean(self):
        """
        The mean of this `EllipticalUniformDistribution`,
        \\(\\boldsymbol{\\mu}\\).
        """
        if not hasattr(self, '_mean'):
            raise AttributeError("mean was referenced before it was set.")
        return self._mean
    
    @mean.setter
    def mean(self, value):
        """
        Setter for `EllipticalUniformDistribution.mean`.
        
        Parameters
        ----------
        value : numpy.ndarray
            1D sequence of numbers describing center of ellipsoid
        """
        if type(value) in sequence_types:
            value = np.array(value)
            if all([(type(element) in numerical_types) for element in value]):
                if len(value) > 1:
                    self._mean = value
                else:
                    raise ValueError("mean is only univariate. Use the " +\
                        "UniformDistribution class instead.")
            else:
                raise TypeError("Not all elements of mean were numbers.")
        else:
            raise TypeError("mean was set to a non-sequence.")
    
    @property
    def covariance(self):
        """
        The covariance of this `EllipticalUniformDistribution`,
        \\(\\boldsymbol{\\Sigma}\\).
        """
        if not hasattr(self, '_covariance'):
            raise AttributeError("covariance was referenced before it was " +\
                "set.")
        return self._covariance
    
    @covariance.setter
    def covariance(self, value):
        """
        Setter for `EllipticalUniformDistribution.covariance`.
        
        Parameters
        ----------
        value : numpy.ndarray
            square positive definite matrix of rank
            `EllipticalUniformDistribution.numparams`
        """
        if type(value) in sequence_types:
            value = np.array(value)
            if value.shape == self.mean.shape:
                self._covariance = np.diag(value)
            elif value.shape == (2 * self.mean.shape):
                self._covariance = value
            else:
                raise ValueError("covariance was neither a vector of " +\
                    "variances nor a matrix of covariances, based on its " +\
                    "shape.")
        else:
            raise TypeError("covariance was set to a non-sequence.")
    
    @property
    def variance(self):
        """
        Alias for `EllipticalUniformDistribution.covariance`.
        """
        return self.covariance
    
    @property
    def square_root_covariance(self):
        """
        The square root of the covariance matrix,
        \\(\\boldsymbol{\\Sigma}^{1/2}\\)
        """
        if not hasattr(self, '_square_root_covariance'):
            self._square_root_covariance = slalg.sqrtm(self.covariance)
        return self._square_root_covariance
    
    @property
    def log_probability(self):
        """
        The logarithm of the probability density inside the ellipse, given by
        \\(\\ln{\\Gamma\\left(\\frac{N}{2}+1\\right)} -\
        \\frac{1}{2}\\ln{\\left|\\boldsymbol{\\Sigma}\\right|}-\
        \\frac{N}{2}\\ln{[(N+2)\\pi]}\\).
        """
        if not hasattr(self, '_log_probability'):
            self._log_probability = log_gamma((self.numparams / 2) + 1) -\
                (lalg.slogdet(self.covariance)[1] / 2.) - (self.numparams *\
                (np.log(np.pi * (self.numparams + 2))) / 2)
        return self._log_probability
    
    @property
    def inverse_covariance(self):
        """
        The inverse of `EllipticalUniformDistribution.covariance`, given by
        \\(\\boldsymbol{\\Sigma}^{-1}\\).
        """
        if not hasattr(self, '_inverse_covariance'):
            self._inverse_covariance = lalg.inv(self.covariance)
        return self._inverse_covariance
    
    @property
    def numparams(self):
        """
        The number of parameters of this `EllipticalUniformDistribution`.
        """
        if not hasattr(self, '_numparams'):
            self._numparams = len(self.mean)
        return self._numparams

    def draw(self, shape=None, random=rand):
        """
        Draws point(s) from this `EllipticalUniformDistribution`. Below, `p` is
        `EllipticalUniformDistribution.numparams`.
        
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
        xis = random.randn(*(shape + (self.numparams,)))
        xis = xis / np.sqrt(np.sum(np.power(xis, 2), axis=-1, keepdims=True))
        # xi now contains random directional unit vectors
        radial_cdfs = random.rand(*shape)
        max_z_radius = np.sqrt(self.numparams + 2)
        fractional_radii = np.power(radial_cdfs, 1. / self.numparams)
        deviations = max_z_radius * fractional_radii[...,np.newaxis] *\
            np.dot(xis, self.square_root_covariance)
        points = self.mean[((np.newaxis,)*len(shape)) + (slice(None),)] +\
            deviations
        if none_shape:
            return points[0]
        else:
            return points
    
    def log_value(self, point):
        """
        Computes the logarithm of the value of this
        `EllipticalUniformDistribution` at the given point.
        
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
        centered_point = np.array(point) - self.mean
        matprod = np.dot(np.dot(centered_point, self.inverse_covariance),\
            centered_point)
        if (matprod <= (self.numparams + 2)):
            return self.log_probability
        else:
            return -np.inf

    def to_string(self):
        """
        Finds and returns a string version of this
        `EllipticalUniformDistribution` of the form `"d-dim elliptical"`.
        """
        return ('{}-dim elliptical'.format(self.numparams))
    
    def __eq__(self, other):
        """
        Checks for equality of this `EllipticalUniformDistribution` with
        `other`.
        
        Parameters
        ----------
        other : object
            object to check for equality
        
        Returns
        -------
        result : bool
            True if and only if `other` is a `EllipticalUniformDistribution`
            defining the same ellipsoid
        """
        if isinstance(other, EllipticalUniformDistribution):
            if self.numparams == other.numparams:
                mean_close =\
                    np.allclose(self.mean, other.mean, rtol=0, atol=1e-9)
                covariance_close = np.allclose(self.covariance,\
                    other.covariance, rtol=1e-12, atol=0)
                metadata_equal = self.metadata_equal(other)
                return all([mean_close, covariance_close, metadata_equal])
            else:
                return False
        else:
            return False
    
    @property
    def minimum(self):
        """
        The minimum allowable value(s) in this distribution.
        """
        # TODO change this to be the actual minimum coordinates of the ellipse!
        return [None] * self.numparams
    
    @property
    def maximum(self):
        """
        The maximum allowable value(s) in this distribution.
        """
        # TODO change this to be the actual maximum coordinates of the ellipse!
        return [None] * self.numparams
    
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
        `EllipticalUniformDistribution` so that it can be loaded later.
        
        Parameters
        ----------
        group : h5py.Group
            hdf5 file group to fill
        mean_link : str or h5py.Dataset or None
            link to existing mean vector in hdf5 file, if it exists
        covariance_link : str or h5py.Dataset or None
            link to existing covariance matrix in hdf5 file, if it exists
        save_metadata : bool
            - if True, attempts to save metadata alongside distribution and
            throws error if it fails
            - if False, metadata is ignored in saving process
        """
        group.attrs['class'] = 'EllipticalUniformDistribution'
        create_hdf5_dataset(group, 'mean', data=self.mean, link=mean_link)
        create_hdf5_dataset(group, 'covariance', data=self.covariance,\
            link=covariance_link)
        if save_metadata:
            self.save_metadata(group)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a `EllipticalUniformDistribution` from the given hdf5 file group.
        
        Parameters
        ----------
        group : h5py.Group
            the same hdf5 file group which fill_hdf5_group was called on when
            this Distribution was saved
        
        Returns
        -------
        distribution : `EllipticalUniformDistribution`
            distribution created from the information in the given group
        """
        try:
            assert group.attrs['class'] == 'EllipticalUniformDistribution'
        except:
            raise TypeError("The given hdf5 file doesn't seem to contain a " +\
                "EllipticalUniformDistribution.")
        metadata = Distribution.load_metadata(group)
        mean = get_hdf5_value(group['mean'])
        covariance = get_hdf5_value(group['covariance'])
        return EllipticalUniformDistribution(mean, covariance,\
            metadata=metadata)
    
    @property
    def gradient_computable(self):
        """
        Boolean describing whether the gradient of the given distribution has
        been implemented. If True,
        `EllipticalUniformDistribution.gradient_of_log_value` method can be
        called safely.
        """
        return True
    
    def gradient_of_log_value(self, point):
        """
        Computes the gradient (derivative) of the logarithm of the value of
        this `EllipticalUniformDistribution` at the given point.
        
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
        return np.zeros((self.numparams,))
    
    @property
    def hessian_computable(self):
        """
        Boolean describing whether the hessian of the given distribution has
        been implemented. If True,
        `EllipticalUniformDistribution.hessian_of_log_value` method can be
        called safely.
        """
        return True
    
    def hessian_of_log_value(self, point):
        """
        Computes the hessian (second derivative) of the logarithm of the value
        of this `EllipticalUniformDistribution` at the given point.
        
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
        return np.zeros((self.numparams,) * 2)
    
    def copy(self):
        """
        Copies this distribution.
        
        Returns
        -------
        copied : `EllipticalUniformDistribution`
            a deep copy of this distribution, ignoring metadata.
        """
        return EllipticalUniformDistribution(self.mean.copy(),\
            self.covariance.copy())

