"""
Module containing class representing a jumping distribution whose destination
is distributed uniformly in an ellipse centered on the source. The PDF of this
distribution is
$$f(\\boldsymbol{x},\\boldsymbol{y})=\\begin{cases}\
\\frac{\\Gamma\\left(\\frac{N}{2}+1\\right)}{\\left|(N+2)\\pi\
\\boldsymbol{\\Sigma}\\right|^{1/2}} & (\\boldsymbol{y}-\\boldsymbol{x})^T\
\\boldsymbol{\\Sigma}^{-1}(\\boldsymbol{y}-\\boldsymbol{x}) \\le N+2 \\\\\
0 & \\text{otherwise} \\end{cases}$$

**File**: $DISTPY/distpy/jumping/UniformJumpingDistribution.py  
**Author**: Keith Tauscher  
**Date**: 11 Jul 2021
"""
import numpy as np
import numpy.linalg as npla
import scipy.linalg as scila
from scipy.special import gammaln as log_gamma
from ..util import int_types, numerical_types, sequence_types,\
    create_hdf5_dataset, get_hdf5_value
from .JumpingDistribution import JumpingDistribution

class UniformJumpingDistribution(JumpingDistribution):
    """
    Class representing a jumping distribution whose destination is distributed
    uniformly in an ellipse centered on the source. The PDF of this
    distribution is $$f(\\boldsymbol{x},\\boldsymbol{y})=\\begin{cases}\
    \\frac{\\Gamma\\left(\\frac{N}{2}+1\\right)}{\\left|(N+2)\\pi\
    \\boldsymbol{\\Sigma}\\right|^{1/2}} & (\\boldsymbol{y}-\\boldsymbol{x})^T\
    \\boldsymbol{\\Sigma}^{-1}(\\boldsymbol{y}-\\boldsymbol{x}) \\le N+2 \\\\\
    0 & \\text{otherwise} \\end{cases}$$
    """
    def __init__(self, covariance):
        """
        Initializes a `UniformJumpingDistribution` with the given covariance
        matrix.
        
        Parameters
        ----------
        covariance : float or numpy.ndarray
            either single number (if this should be a 1D uniform) or square 2D
            array (if this should be a multivariate ellipse)
        """
        self.covariance = covariance
    
    @property
    def covariance(self):
        """
        A 2D `numpy.ndarray` of covariances.
        """
        if not hasattr(self, '_covariance'):
            raise AttributeError("covariance referenced before it was set.")
        return self._covariance
    
    @covariance.setter
    def covariance(self, value):
        """
        Setter for `UniformJumpingDistribution.covariance`
        
        Parameters
        ----------
        value : float or numpy.ndarray
            either a single number (if this should be 1D) or a square 2D array
        """
        if type(value) in numerical_types:
            self._covariance = np.ones((1, 1)) * value
        elif type(value) in sequence_types:
            value = np.array(value)
            if (value.ndim == 2) and (value.shape[0] == value.shape[1]):
                self._covariance = value
            else:
                raise ValueError("covariance didn't have the expected shape.")
        else:
            raise TypeError("covariance was neither a number nor an array.")
        self.inverse_covariance, self.constant_log_value # compute stuff
    
    @property
    def inverse_covariance(self):
        """
        A 2D numpy.ndarray storing the inverse of
        `UniformJumpingDistribution.covariance`
        """
        if not hasattr(self, '_inverse_covariance'):
            self._inverse_covariance = npla.inv(self.covariance)
        return self._inverse_covariance
    
    @property
    def constant_log_value(self):
        """
        A constant in the log value which is independent of both the source and
        the destination.
        """
        if not hasattr(self, '_constant_log_value'):
            n_over_2 = self.numparams / 2.
            n_plus_2 = self.numparams + 2
            self._constant_log_value = log_gamma(n_over_2 + 1) -\
                (n_over_2 * (np.log(np.pi * (n_plus_2)))) -\
                (npla.slogdet(self.covariance)[1] / 2.)
        return self._constant_log_value
    
    @property
    def matrix_for_draw(self):
        """
        Property storing the matrix square root of
        `self.covariance * (self.numparams + 2)`, which plays an important role
        in the efficient drawing from this `UniformJumpingDistribution`.
        """
        return scila.sqrtm(self.covariance * (self.numparams + 2))
    
    def draw(self, source, shape=None, random=np.random):
        """
        Draws a destination point from this `UniformJumpingDistribution` given
        a source point.
        
        Parameters
        ----------
        source : number or numpy.ndarray
            - if this JumpingDistribution is univariate, source should be
            a single number
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
            return random.uniform(source - self.half_span,\
                source + self.half_span, size=shape)
        else:
            none_shape = (type(shape) is type(None))
            if none_shape:
                shape = (1,)
            elif type(shape) in int_types:
                shape = (shape,)
            normal_vector =\
                random.standard_normal(size=shape+(self.numparams,))
            radii = np.power(random.random(size=shape), 1. / self.numparams)
            radii = (radii / npla.norm(normal_vector, axis=-1))[...,np.newaxis]
            displacement = radii * np.dot(normal_vector, self.matrix_for_draw)
            destination = displacement +\
                source[((np.newaxis,)*len(shape))+(slice(None),)]
            if none_shape:
                return destination[0]
            else:
                return destination
    
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
        chi2 = np.dot(difference, np.dot(difference, self.inverse_covariance))
        if chi2 < (self.numparams + 2):
            return self.constant_log_value
        else:
            return -np.inf
    
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
            0, indicating that a jump
            \\(\\boldsymbol{x}\\rightarrow\\boldsymbol{y}\\) is equally likely
            as a jump \\(\\boldsymbol{y}\\rightarrow\\boldsymbol{x}\\)
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
    def half_span(self):
        """
        The distance between the source and farthest possible destination
        (valid only in the case where this distribution is univariate!).
        """
        if not hasattr(self, '_half_span'):
            if self.numparams == 1:
                self._half_span = np.sqrt(self.covariance[0,0] * 3)
            else:
                raise NotImplementedError("The span of a multivariate " +\
                    "distribution is not well-defined and thus can't be " +\
                    "referenced.")
        return self._half_span
    
    def __eq__(self, other):
        """
        Tests for equality between this distribution and other.
        
        Parameters
        ----------
        other : object
            object with which to check for equality
        
        Returns
        -------
        result : bool
            True if and only if `other` is another `UniformJumpingDistribution`
            with the same covariance
        """
        if isinstance(other, UniformJumpingDistribution):
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
        Boolean (False) describing whether this distribution describes discrete
        (True) or continuous (False) variable(s).
        """
        return False
    
    def fill_hdf5_group(self, group, covariance_link=None):
        """
        Fills the given hdf5 file group with data from this distribution.
        
        Parameters
        ----------
        group : h5py.Group
            hdf5 file group to fill
        covariance_link : str or h5py.Dataset or None
            link to mean vector in hdf5 file, if it exists
        """
        group.attrs['class'] = 'UniformJumpingDistribution'
        create_hdf5_dataset(group, 'covariance', data=self.covariance,\
            link=covariance_link)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a `UniformJumpingDistribution` from the given hdf5 file group.
        
        Parameters
        ----------
        group : h5py.Group
            the same hdf5 file group which
            `UniformJumpingDistribution.fill_hdf5_group` was called on
        
        Returns
        -------
        loaded : `UniformJumpingDistribution`
            a `UniformJumpingDistribution` object created from the information
            in the given group
        """
        try:
            assert group.attrs['class'] == 'UniformJumpingDistribution'
        except:
            raise ValueError("The given group does not seem to contain a " +\
                "UniformJumpingDistribution.")
        return UniformJumpingDistribution(get_hdf5_value(group['covariance']))

