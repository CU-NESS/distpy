"""
Module containing class representing a Gaussian jumping distribution. Its PDF
is given by $$f(\\boldsymbol{x},\\boldsymbol{y})=\
\\left| 2\\pi\\boldsymbol{\\Sigma}\\right|^{-1/2}\\ \\exp{\\left\\{\
-\\frac{1}{2}(\\boldsymbol{y}-\\boldsymbol{x})^T\\boldsymbol{\\Sigma}^{-1}\
(\\boldsymbol{y}-\\boldsymbol{x})\\right\\}}$$

**File**: $DISTPY/distpy/jumping/GaussianJumpingDistribution.py  
**Author**: Keith Tauscher  
**Date**: 3 Jul 2021
"""
import numpy as np
import numpy.linalg as npla
import scipy.linalg as scila
from ..util import create_hdf5_dataset, get_hdf5_value, int_types,\
    numerical_types, sequence_types
from .JumpingDistribution import JumpingDistribution

class GaussianJumpingDistribution(JumpingDistribution):
    """
    Class representing a Gaussian jumping distribution. Its PDF is given by
    $$f(\\boldsymbol{x},\\boldsymbol{y})= \\left| 2\\pi\\boldsymbol{\\Sigma}\
    \\right|^{-1/2}\\ \\exp{\\left\\{-\\frac{1}{2}(\\boldsymbol{y}-\
    \\boldsymbol{x})^T\\boldsymbol{\\Sigma}^{-1}\
    (\\boldsymbol{y}-\\boldsymbol{x})\\right\\}}$$
    """
    def __init__(self, covariance):
        """
        Initializes a `GaussianJumpingDistribution` with the given covariance
        matrix.
        
        Parameters
        ----------
        covariance : float or numpy.ndarray
            either single number (if this should be a 1D Gaussian) or square
            2D array (if this should be a multivariate Gaussian)
        """
        self.covariance = covariance
    
    @property
    def covariance(self):
        """
        A 2D numpy.ndarray of covariances.
        """
        if not hasattr(self, '_covariance'):
            raise AttributeError("covariance referenced before it was set.")
        return self._covariance
    
    @covariance.setter
    def covariance(self, value):
        """
        Setter for `GaussianJumpingDistribution.covariance`.
        
        Parameters
        ----------
        value : float or numpy.ndarray
            either single number (if this should be a 1D Gaussian) or square
            2D array (if this should be a multivariate Gaussian)
        """
        if type(value) in numerical_types:
            self._covariance = np.ones((1, 1)) * value
        elif type(value) in sequence_types:
            value = np.array(value)
            if np.any(np.isnan(value)):
                raise ValueError(("For some reason, there are nan's in the " +\
                    "covariance matrix given to a " +\
                    "GaussianJumpingDistribution, which was:\n{}.").format(\
                    value))
            elif (value.ndim == 2) and (value.shape[0] == value.shape[1]):
                self._covariance = (value + value.T) / 2
            else:
                raise ValueError("covariance didn't have the expected shape.")
        else:
            raise TypeError("covariance was neither a number nor an array.")
        self.inverse_covariance, self.constant_in_log_value # compute stuff
    
    @property
    def inverse_covariance(self):
        """
        A 2D numpy.ndarray storing the inverse of
        `GaussianJumpingDistribution.covariance`.
        """
        if not hasattr(self, '_inverse_covariance'):
            self._inverse_covariance = npla.inv(self.covariance)
        return self._inverse_covariance
    
    @property
    def constant_in_log_value(self):
        """
        A constant in the log value which is independent of both the source and
        the destination.
        """
        if not hasattr(self, '_constant_in_log_value'):
            self._constant_in_log_value =\
                ((self.numparams * np.log(2 * np.pi)) +\
                npla.slogdet(self.covariance)[1]) / (-2.)
        return self._constant_in_log_value
    
    @property
    def square_root_covariance(self):
        """
        The square root of `GaussianJumpingDistribution.covariance`.
        """
        if not hasattr(self, '_square_root_covariance'):
            (eigenvalues, eigenvectors) = npla.eigh(self.covariance)
            if np.any(eigenvalues <= 0):
                raise ValueError(("Something went wrong, causing the square " +\
                    "root of the covariance matrix of this " +\
                    "GaussianJumpingDistribution to have at least one " +\
                    "complex element. The eigenvalues of the covariance " +\
                    "matrix are {!s}.").format(eigenvalues))
            eigenvalues = np.sqrt(eigenvalues)
            self._square_root_covariance =\
                np.dot(eigenvectors * eigenvalues[None,:], eigenvectors.T)
        return self._square_root_covariance
    
    def draw(self, source, shape=None, random=np.random):
        """
        Draws a destination point from this jumping distribution given a source
        point.
        
        Parameters
        ----------
        source : number or numpy.ndarray
            - if this `GaussianJumpingDistribution` is univariate, source
            should be a single number
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
        drawn : number or numpy.ndarray
            either single value or array of values. See documentation on
            `shape` above for the type of the returned value
        """
        if self.numparams == 1:
            return random.normal(source, self.standard_deviation, size=shape)
        else:
            if type(shape) is type(None):
                shape = ()
            if type(shape) in int_types:
                shape = (shape,)
            return source[((np.newaxis,) * len(shape)) + (slice(None),)] +\
                np.dot(random.normal(0, 1, size=shape+(self.numparams,)),\
                self.square_root_covariance)
    
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
            if the distribution is \\(f(\\boldsymbol{x},\\boldsymbol{y})=\
            \\text{Pr}[\\boldsymbol{y}|\\boldsymbol{x}]\\), `source` is
            \\(\\boldsymbol{x}\\) and `destination` is \\(\\boldsymbol{y}\\),
            then `log_pdf` is given by
            \\(\\ln{f(\\boldsymbol{x},\\boldsymbol{y})}\\)
        """
        difference = (destination - source)
        if self.numparams == 1:
            return (self.constant_in_log_value +\
                (((difference / self.standard_deviation) ** 2) / (-2.)))
        else:
            return (self.constant_in_log_value + (np.dot(difference,\
                np.dot(difference, self.inverse_covariance)) / (-2.)))
    
    def log_value_difference(self, source, destination):
        """
        Computes the difference in the log-PDF of jumping from `source` to
        `destination` and of jumping from `destination` to `source`. While this
        method has a default version, overriding it may provide an efficiency
        benefit.
        
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
            `log_pdf_difference` will always be zero because
            `GaussianJumpingDistribution` objects assign the same probability
            of jumping from \\(\\boldsymbol{x}\\rightarrow\\boldsymbol{y}\\) to
            jumping from \\(\\boldsymbol{y}\\rightarrow\\boldsymbol{x}\\)
        """
        return 0.
    
    @property
    def numparams(self):
        """
        The integer number of parameters described by this distribution.
        """
        if not hasattr(self, '_numparams'):
            self._numparams = self.covariance.shape[0]
        return self._numparams
    
    @property
    def standard_deviation(self):
        """
        The square root of the variance (in the case that
        `GaussianJumpingDistribution.numparams` == 1). If this distribution is
        multivariate, referencing this property will throw a
        `NotImplementedError` because the standard deviation is not well
        defined in this case.
        """
        if not hasattr(self, '_standard_deviation'):
            if self.numparams == 1:
                self._standard_deviation = np.sqrt(self.covariance[0,0])
            else:
                raise NotImplementedError("The standard deviation of a " +\
                    "multivariate Gaussian was referenced, but the " +\
                    "standard deviation has no well defined meaning for " +\
                    "multivariate Gaussian distributions.")
        return self._standard_deviation
    
    def __eq__(self, other):
        """
        Tests for equality between this `GaussianJumpingDistribution` and
        `other`.
        
        Parameters
        ----------
        other : object
            object with which to check for equality
        
        Returns
        -------
        result : bool
            True if and only if object is a `GaussianJumpingDistribution` with
            the same covariance matrix
        """
        if isinstance(other, GaussianJumpingDistribution):
            if self.numparams == other.numparams:
                return np.allclose(self.covariance, other.covariance,\
                    rtol=1e-12, atol=1e-12)
            else:
                return False
        else:
            return False
    
    @property
    def is_discrete(self):
        """
        Boolean describing whether this JumpingDistribution describes discrete
        (True) or continuous (False) variable(s). Since this is a continuous
        distribution, it is always False.
        """
        return False
    
    def fill_hdf5_group(self, group, covariance_link=None):
        """
        Fills the given hdf5 file group with data from this distribution.
        
        Parameters
        ----------
        group : h5py.Group
            hdf5 file group to fill
        """
        group.attrs['class'] = 'GaussianJumpingDistribution'
        create_hdf5_dataset(group, 'covariance', data=self.covariance,\
            link=covariance_link)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a `GaussianJumpingDistribution` from the given hdf5 file group.
        
        Parameters
        ----------
        group : h5py.Group
            the same hdf5 file group which
            `GaussianJumpingDistribution.fill_hdf5_group` was called on
        
        Returns
        -------
        loaded : `GaussianJumpingDistribution`
            a `GaussianJumpingDistribution` object loaded from the given group
        """
        try:
            assert group.attrs['class'] == 'GaussianJumpingDistribution'
        except:
            raise ValueError("The given group does not seem to contain a " +\
                "GaussianJumpingDistribution.")
        return GaussianJumpingDistribution(get_hdf5_value(group['covariance']))

