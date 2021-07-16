"""
Module containing class representing a jumping distribution whose translation
is Gaussian distributed with a covariance depending on the source. Its PDF is
given by $$f(\\boldsymbol{x},\\boldsymbol{y})=\
\\left|2\\pi\\boldsymbol{\\Sigma}(\\boldsymbol{x})\\right|^{-1/2}\
\\exp{\\left\\{-\\frac{1}{2}(\\boldsymbol{y}-\\boldsymbol{x})^T\
\\left[\\boldsymbol{\\Sigma}(\\boldsymbol{x})\\right]^{-1}\
(\\boldsymbol{y}-\\boldsymbol{x})\\right\\}},$$ where
\\(\\boldsymbol{\\Sigma}(\\boldsymbol{x})\\) is determined from a set of points
\\(\\boldsymbol{z}_k\\) and associated covariances
\\(\\boldsymbol{\\Lambda}_k\\). In particular,
\\(\\boldsymbol{\\Sigma}(\\boldsymbol{x})=\\boldsymbol{\\Lambda}_{\\kappa}\\),
where \\(\\kappa=\\underset{k}{\\text{argmin}}\\left\\{\
(\\boldsymbol{x}-\\boldsymbol{z}_k)^T\
\\left[\\boldsymbol{\\Lambda}_k\\right]^{-1}\
(\\boldsymbol{x}-\\boldsymbol{z}_k)\\right\\}\\).

**File**: $DISTPY/distpy/jumping/TruncatedGaussianJumpingDistribution.py  
**Author**: Keith Tauscher  
**Date**: 11 Jul 2021
"""
import numpy as np
import numpy.linalg as npla
from ..util import int_types, numerical_types, sequence_types,\
    create_hdf5_dataset, get_hdf5_value
from .JumpingDistribution import JumpingDistribution

class SourceDependentGaussianJumpingDistribution(JumpingDistribution):
    """
    Class representing a jumping distribution whose translation is Gaussian
    distributed with a covariance depending on the source. Its PDF is given by
    $$f(\\boldsymbol{x},\\boldsymbol{y})=\
    \\left|2\\pi\\boldsymbol{\\Sigma}(\\boldsymbol{x})\\right|^{-1/2}\
    \\exp{\\left\\{-\\frac{1}{2}(\\boldsymbol{y}-\\boldsymbol{x})^T\
    \\left[\\boldsymbol{\\Sigma}(\\boldsymbol{x})\\right]^{-1}\
    (\\boldsymbol{y}-\\boldsymbol{x})\\right\\}},$$ where
    \\(\\boldsymbol{\\Sigma}(\\boldsymbol{x})\\) is determined from a set of
    points \\(\\boldsymbol{z}_k\\) and associated covariances
    \\(\\boldsymbol{\\Lambda}_k\\). In particular,
    \\(\\boldsymbol{\\Sigma}(\\boldsymbol{x})=\
    \\boldsymbol{\\Lambda}_{\\kappa}\\),
    where \\(\\kappa=\\underset{k}{\\text{argmin}}\\left\\{\
    (\\boldsymbol{x}-\\boldsymbol{z}_k)^T\
    \\left[\\boldsymbol{\\Lambda}_k\\right]^{-1}\
    (\\boldsymbol{x}-\\boldsymbol{z}_k)\\right\\}\\).
    """
    def __init__(self, points, covariances):
        """
        Initializes a `SourceDependentGaussianJumpingDistribution` with the
        given points and covariance matrices.
        
        Parameters
        ----------
        points : sequence
            sequence of either single numbers (if this should be a 1D Gaussian)
            or arrays of length numparams (if this should be a multivariate
            Gaussian), representing \\(\\boldsymbol{z}_k\\)
        covariances : sequence
            sequence of either single numbers (if this should be a 1D Gaussian)
            or square 2D arrays (if this should be a multivariate Gaussian),
            representing \\(\\boldsymbol{\\Sigma}_k\\)
        """
        self.points = points
        self.covariances = covariances
    
    @property
    def points(self):
        """
        The points, \\(\\boldsymbol{z}_k\\), which define the centers of the
        different covariance regions.
        """
        if not hasattr(self, '_points'):
            raise AttributeError("points was referenced before it was set.")
        return self._points
    
    @points.setter
    def points(self, value):
        """
        Setter for `SourceDependentGaussianJumpingDistribution.points`.
        
        Parameters
        ----------
        value : sequence
            sequence of either single numbers (if this should be a 1D Gaussian)
            or arrays of length numparams (if this should be a multivariate
            Gaussian), representing \\(\\boldsymbol{z}_k\\)
        """
        if type(value) in sequence_types:
            if all([type(element) in numerical_types for element in value]):
                self._points = [(element * np.ones(1)) for element in value]
            elif all([type(element) in sequence_types for element in value]):
                value = [np.array(element) for element in value]
                if all([element.shape == value[0].shape for element in value]):
                    if value[0].ndim == 1:
                        self._points = [element.copy() for element in value]
                    else:
                        raise ValueError("Array elements of points " +\
                            "sequence were not 1-dimensional.")
                else:
                    raise ValueError("Not all array elements of points " +\
                        "sequence had the same shape.")
            else:
                raise TypeError("Neither all elements of points were " +\
                    "numbers nor all elements of points were sequences.")
        else:
            raise TypeError("points was set to a non-sequence.")
    
    @property
    def num_points(self):
        """
        The number of different regions/number of points.
        """
        if not hasattr(self, '_num_points'):
            self._num_points = len(self.points)
        return self._num_points
    
    @property
    def covariances(self):
        """
        The sequence of 2D numpy.ndarray of covariances,
        \\(\\boldsymbol{\\Sigma}_k\\).
        """
        if not hasattr(self, '_covariances'):
            raise AttributeError("covariances referenced before it was set.")
        return self._covariances
    
    @covariances.setter
    def covariances(self, value):
        """
        Setter for `SourceDependentGaussianJumpingDistribution.covariances`.
        
        Parameters
        ----------
        value : sequence
            sequence of either single numbers (if this should be a 1D Gaussian)
            or square 2D arrays (if this should be a multivariate Gaussian),
            representing \\(\\boldsymbol{\\Sigma}_k\\)
        """
        if type(value) in sequence_types:
            if all([type(element) in numerical_types for element in value]):
                self._covariances =\
                    [(np.ones((1, 1)) * element) for element in value]
            elif all([type(element) in sequence_types for element in value]):
                value = [np.array(element) for element in value]
                if any([np.any(np.isnan(element)) for element in value]):
                    raise ValueError("For some reason, there are nan's in " +\
                        "a covariance matrix given to a " +\
                        "SourceDependentGaussianJumpingDistribution.")
                elif all([(element.shape == value[0].shape)\
                    for element in value]):
                    if value[0].shape == ((self.numparams,) * 2):
                        self._covariances = [((element + element.T) / 2)\
                            for element in value]
                    else:
                        raise ValueError("Array elements of covariance " +\
                            "matrices did not have the expected shape.")
                else:
                    raise ValueError("Not all array elements of covariance " +\
                        "matrices had the same shape.")
            else:
                raise TypeError("Neither all covariances were single " +\
                    "numbers nor all covariances were sequences.")
        else:
            raise TypeError("covariances was set to a non-sequence.")
        self.inverse_covariances
        self.square_root_covariances
        self.constants_in_log_value
    
    @property
    def inverse_covariances(self):
        """
        Sequence of 2D `numpy.ndarray` objects storing the inverses of the
        matrices stored in
        `SourceDependentGaussianJumpingDistribution.covariances` property.
        """
        if not hasattr(self, '_inverse_covariances'):
            self._inverse_covariances =\
                [npla.inv(covariance) for covariance in self.covariances]
        return self._inverse_covariances
    
    @property
    def constants_in_log_value(self):
        """
        A constant in the log value which is independent of both the source and
        the destination.
        """
        if not hasattr(self, '_constants_in_log_value'):
            self._constants_in_log_value =\
                [((self.numparams * np.log(2 * np.pi)) +\
                npla.slogdet(covariance)[1]) / (-2.)\
                for covariance in self.covariances]
        return self._constants_in_log_value
    
    @property
    def square_root_covariances(self):
        """
        Sequence of 2D `numpy.ndarray` objects storing the square roots of the
        matrices stored in
        `SourceDependentGaussianJumpingDistribution.covariances` property.
        """
        if not hasattr(self, '_square_root_covariances'):
            self._square_root_covariances = []
            for covariance in self.covariances:
                (eigenvalues, eigenvectors) = npla.eigh(covariance)
                if np.any(eigenvalues <= 0):
                    raise ValueError(("Something went wrong, causing the " +\
                        "square root of a covariance matrix of this " +\
                        "SourceDependentGaussianJumpingDistribution to " +\
                        "have at least one complex element. The " +\
                        "eigenvalues of the covariance matrix are " +\
                        "{!s}.").format(eigenvalues))
                eigenvalues = np.sqrt(eigenvalues)
                self._square_root_covariances.append(\
                    np.dot(eigenvectors * eigenvalues[None,:], eigenvectors.T))
        return self._square_root_covariances
    
    @property
    def standard_deviations(self):
        """
        The standard deviations possible for this
        `SourceDependentGaussianJumpingDistribution` if it is univariate.
        """
        if not hasattr(self, '_standard_deviations'):
            if self.numparams == 1:
                self._standard_deviations = [np.sqrt(covariance[0,0])\
                    for covariance in self.covariances]
            else:
                raise NotImplementedError("standard_deviations not defined " +\
                    "because the distribution is not 1D.")
        return self._standard_deviations
    
    def choose_point_from_source(self, source):
        """
        Finds the index of the point, \\(\\boldsymbol{z}_k\\), defining the
        region whose covariance should be used when jumping from the given
        source.
        
        Parameters
        ----------
        source : number or numpy.ndarray
            single number if univariate, 1D array if multivariate
        
        Returns
        -------
        region : int
            integer index of point to use to define distribution from `source`
        """
        return np.argmin([np.dot(pnt - source, np.dot(pnt - source, invcov))\
            for (pnt, invcov) in zip(self.points, self.inverse_covariances)])
    
    def draw(self, source, shape=None, random=np.random):
        """
        Draws a destination point from this jumping distribution given a source
        point.
        
        Parameters
        ----------
        source : number or numpy.ndarray
            - if this `SourceDependentGaussianJumpingDistribution` is
            univariate, source should be a single number
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
        destinations : number or numpy.ndarray
            either single value or array of values. See documentation on
            `shape` above for the type of the returned value
        """
        point_index = self.choose_point_from_source(source)
        if self.numparams == 1:
            return random.normal(source,\
                self.standard_deviations[point_index], size=shape)
        elif type(shape) is type(None):
            return source + np.dot(self.square_root_covariances[point_index],\
                random.normal(0, 1, size=self.numparams))
        else:
            if type(shape) in int_types:
                shape = (shape,)
            random_vector = random.normal(0, 1, size=shape+(1, self.numparams))
            return source + np.sum(random_vector *\
                self.square_root_covariances[point_index], axis=-1)
    
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
        point_index = self.choose_point_from_source(source)
        difference = (destination - source)
        if self.numparams == 1:
            return (self.constants_in_log_value[point_index] +\
                (((difference / self.standard_deviations[point_index]) ** 2) /\
                (-2.)))
        else:
            return (self.constants_in_log_value[point_index] +\
                (np.dot(difference, np.dot(difference,\
                self.inverse_covariances[point_index])) / (-2.)))
    
    @property
    def numparams(self):
        """
        The integer number of parameters described by this distribution.
        """
        if not hasattr(self, '_numparams'):
            self._numparams = len(self.points[0])
        return self._numparams
    
    def __eq__(self, other):
        """
        Tests for equality between this
        `SourceDependentGaussianJumpingDistribution` and `other`.
        
        Parameters
        ----------
        other : object
            object with which to check for equality
        
        Returns
        -------
        result : bool
            True if and only if object is another
            `SourceDependentGaussianJumpingDistribution` with the same
            `SourceDependentGaussianJumpingDistribution.points` and
            `SourceDependentGaussianJumpingDistribution.covariances`
        """
        if isinstance(other, SourceDependentGaussianJumpingDistribution):
            if self.numparams == other.numparams:
                if self.num_points == other.num_points:
                    points_equal = [np.allclose(*points)\
                        for points in zip(self.points, other.points)]
                    covariances_equal = [np.allclose(*covs)\
                        for covs in zip(self.covariances, other.covariances)]
                    return (points_equal and covariances_equal)
                else:
                    return False
            else:
                return False
        else:
            return False
    
    @property
    def is_discrete(self):
        """
        Toolean describing whether this distribution describes discrete (True)
        or continuous (False) variable(s). Since Gaussian distributions are
        continuous, this is always False.
        """
        return False
    
    def fill_hdf5_group(self, group, covariance_link=None):
        """
        Fills the given hdf5 file group with data from this distribution.
        
        Parameters
        ----------
        group : h5py.Group
            hdf5 file group to fill with information about this distribution
        """
        group.attrs['class'] = 'SourceDependentGaussianJumpingDistribution'
        create_hdf5_dataset(group, 'points', data=self.points)
        create_hdf5_dataset(group, 'covariances', data=self.covariances)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a `SourceDependentGaussianJumpingDistribution` from the given
        hdf5 file group.
        
        Parameters
        ----------
        group : h5py.Group
            the same hdf5 file group which
            `SourceDependentGaussianJumpingDistribution.fill_hdf5_group` was
            called on
        
        Returns
        -------
        loaded : `SourceDependentGaussianJumpingDistribution
            distribution loaded from information in the given group
        """
        try:
            assert(group.attrs['class'] ==\
                'SourceDependentGaussianJumpingDistribution')
        except:
            raise ValueError("The given group does not seem to contain a " +\
                "SourceDependentGaussianJumpingDistribution.")
        points = [point for point in get_hdf5_value(group['points'])]
        covariances =\
            [covariance for covariance in get_hdf5_value(group['covariances'])]
        return SourceDependentGaussianJumpingDistribution(points, covariances)

