"""
Module containing class representing a distribution that is uniform over the
convex hull of an arbitrary set of points.

**File**: $DISTPY/distpy/distribution/UniformTriangulationDistribution.py  
**Author**: Keith Tauscher  
**Date**: 31 May 2021
"""
import numpy as np
import numpy.linalg as la
import numpy.random as rand
from scipy.spatial import Delaunay
from ..util import int_types, create_hdf5_dataset, get_hdf5_value
from .Distribution import Distribution

class UniformTriangulationDistribution(Distribution):
    """
    Class representing a distribution that is uniform over the convex hull of
    an arbitrary set of points.
    """
    def __init__(self, triangulation=None, points=None, metadata=None):
        """
        Initializes a new `UniformTriangulationDistribution` with the given
        parameter values.
        
        Parameters
        ----------
        triangulation : `scipy.spatial.Delaunay` or None
            triangulation to use if it already exists. Can only be None if
            `points` is not None
        points : `numpy.ndarray` or None
            array of shape \\((N_{\\text{points}},N_{\\text{dim}})\\)
            containing the points whose convex hull determines the support of
            this distribution. Can only be None if `triangulation` is not None
        metadata : number or str or dict or `distpy.util.Savable.Savable`
            data to store alongside this distribution.
        """
        self.triangulation = (triangulation, points)
        self.metadata = metadata
    
    @property
    def triangulation(self):
        """
        The triangulation at the heart of this distribution.
        """
        if not hasattr(self, '_triangulation'):
            raise AttributeError("triangulation was referenced before it " +\
                "was set.")
        return self._triangulation
    
    @triangulation.setter
    def triangulation(self, value):
        """
        Setter for `UniformTriangulationDistribution.triangulation`.
        
        Parameters
        ----------
        value : tuple
            tuple of form `(triangulation, points)` where one and only one of
            these may be None.
        """
        if type(value[0]) is type(None):
            if type(value[1]) is type(None):
                raise ValueError("If triangulation is not given, points " +\
                    "must be given. Neither were given.")
            else:
                self._triangulation = Delaunay(value[1])
        else:
            self._triangulation = value[0]

    @property
    def numparams(self):
        """
        The number of parameters of this `UniformTriangulationDistribution`.
        """
        if not hasattr(self, '_numparams'):
            self._numparams = self.triangulation.points.shape[1]
        return self._numparams
    
    @property
    def mean(self):
        """
        The mean of the `UniformTriangulationDistribution` class is not
        implemented.
        """
        if not hasattr(self, '_mean'):
            raise AttributeError("mean is not implemented for the " +\
                "UniformTriangulationDistribution class.")
        return self._mean
    
    @property
    def variance(self):
        """
        The variance of the `UniformTriangulationDistribution` class is not
        implemented.
        """
        if not hasattr(self, '_variance'):
            raise AttributeError("variance is not implemented for the " +\
                "UniformTriangulationDistribution class.")
        return self._variance
    
    @property
    def nsimplices(self):
        """
        The number of simplices in the triangulation at the heart of this
        distribution.
        """
        if not hasattr(self, '_nsimplices'):
            self._nsimplices = len(self.triangulation.simplices)
        return self._nsimplices

    @property
    def numparams_factorial(self):
        """
        The factorial of the number of parameters.
        """
        if not hasattr(self, '_numparams_factorial'):
            self._numparams_factorial = np.prod(np.arange(self.numparams) + 1)
        return self._numparams_factorial
    
    @property
    def simplex_volumes(self):
        """
        The volumes of the simplices in the Triangulation at the heart of this
        `UniformTriangulationDistribution`.
        """
        if not hasattr(self, '_simplex_volumes'):
            parallelotope_volumes = np.ndarray((self.nsimplices,))
            for (isimplex, simplex) in enumerate(self.triangulation.simplices):
                vertices = self.triangulation.points[simplex]
                vertices = vertices[1:,:] - vertices[:1,:]
                parallelotope_volume = np.abs(la.det(vertices))
                parallelotope_volumes[isimplex] = parallelotope_volume
            self._simplex_volumes =\
                parallelotope_volumes / self.numparams_factorial
        return self._simplex_volumes
    
    @property
    def total_volume(self):
        """
        The volume of the convex hull of the simplices in the triangulation.
        """
        if not hasattr(self, '_total_volume'):
            self.discrete_cdf
        return self._total_volume
    
    @property
    def discrete_cdf(self):
        """
        The discrete cdf which describes the relative weighting of the
        simplices and allows us to draw uniformly even though simplices have
        different volumes.
        """
        if not hasattr(self, '_discrete_cdf'):
            self._discrete_cdf = np.cumsum(self.simplex_volumes)
            self._total_volume = self._discrete_cdf[-1]
            self._discrete_cdf = self._discrete_cdf / self.total_volume
        return self._discrete_cdf
    
    def draw_coefficients(self, random=rand):
        """
        Draws numparams+1 numbers which are all in [0, 1] and add up to 1. It
        samples them uniformly.
        
        NOTE: This algorithm is O(N log(N)) time. O(N) can be achieved by
              sampling from an exponential distribution and summing and
              normalizing.
        
        Parameters
        ----------
        random : numpy.random.RandomState
            the random number generator to use
        
        Returns
        -------
        coefficients : numpy.ndarray
            array of `UniformTriangulationDistribution.numparams`+1 numbers
            which are all in \\([0,1]\\) and add up to 1.
        """
        temp_space = random.uniform(size=self.numparams)
        return np.diff(np.sort(np.concatenate([[0], temp_space, [1]])))
    
    def draw_from_simplex(self, isimplex, random=rand):
        """
        Draws a point uniformly from the simplex associated with the given
        index.
        
        Parameters
        ----------
        isimplex : int
            index associated with the simplex to sample
        random : numpy.random.RandomState
            the random number generator to use
        
        Returns
        -------
        point : numpy.ndarray
            a single point from the given simplex
        """
        vertices =\
            self.triangulation.points[self.triangulation.simplices[isimplex],:]
        return np.dot(self.draw_coefficients(random=random), vertices)

    def draw(self, shape=None, random=rand):
        """
        Draws point(s) from this `UniformTriangulationDistribution`. Below, `p`
        is `UniformTriangulationDistribution.numparams`.
        
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
        if type(shape) is type(None):
            shape = 1
        int_shape = type(shape) in int_types
        if int_shape:
            shape = (shape,)
        total_number = np.prod(shape)
        samples = np.ndarray((total_number, self.numparams))
        cdf_values = random.uniform(size=total_number)
        for index in range(total_number):
            cdf_value = cdf_values[index]
            isimplex = np.where(self.discrete_cdf >= cdf_value)[0][0]
            samples[index,:] = self.draw_from_simplex(isimplex, random=random)
        if int_shape and (shape == (1,)):
            return samples[0]
        else:
            return np.reshape(samples, shape + (self.numparams,))

    
    @property
    def constant_log_value(self):
        """
        The log of the value of this distribution inside the convex hull of the
        given points.
        """
        if not hasattr(self, '_constant_log_value'):
            self._constant_log_value = -np.log(self.total_volume)
        return self._constant_log_value

    def log_value(self, point):
        """
        Computes the logarithm of the value of this
        `UniformTriangulationDistribution` at the given point.
        
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
        if self.triangulation.find_simplex(point) == -1:
            return -np.inf
        else:
            return self.constant_log_value
    
    def to_string(self):
        """
        Finds and returns a string version of this
        `UniformTriangulationDistribution` of the form
        `"UniformTriangulationDistribution"`.
        """
        return "UniformTriangulationDistribution"
    
    def __eq__(self, other):
        """
        Checks for equality of this `UniformTriangulationDistribution` with
        `other`.
        
        Parameters
        ----------
        other : object
            object to check for equality
        
        Returns
        -------
        result : bool
            True if and only if `other` is a `UniformTriangulationDistribution`
            with the same points and simplices
        """
        if isinstance(other, UniformTriangulationDistribution):
            tol_kwargs = {'rtol': 1e-6, 'atol': 1e-6}
            points_close = np.allclose(self.triangulation.points,\
                other.triangulation.points, **tol_kwargs)
            simplices_close = np.allclose(self.triangulation.simplices,\
                other.triangulation.simplices)
            metadata_equal = self.metadata_equal(other)
            return all([points_close, simplices_close, metadata_equal])
        else:
            return False
    
    @property
    def minimum(self):
        """
        The minimum allowable value(s) in this distribution.
        """
        # TODO make this actual minimum coordinates of convex hull
        return [None] * self.numparams
    
    @property
    def maximum(self):
        """
        The maximum allowable value(s) in this distribution.
        """
        # TODO make this actual maximum coordinates of convex hull
        return [None] * self.numparams
    
    @property
    def is_discrete(self):
        """
        Boolean describing whether this distribution is discrete (True) or
        continuous (False).
        """
        return False
    
    def fill_hdf5_group(self, group, save_metadata=True):
        """
        Fills the given hdf5 file group with data about this
        `UniformTriangulationDistribution` so that it can be loaded later.
        
        Parameters
        ----------
        group : h5py.Group
            hdf5 file group to fill
        save_metadata : bool
            - if True, attempts to save metadata alongside distribution and
            throws error if it fails
            - if False, metadata is ignored in saving process
        """
        group.attrs['class'] = 'UniformTriangulationDistribution'
        create_hdf5_dataset(group, 'points', data=self.triangulation.points)
        if save_metadata:
            self.save_metadata(group)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a `UniformTriangulationDistribution` from the given hdf5 file
        group.
        
        Parameters
        ----------
        group : h5py.Group
            the same hdf5 file group which fill_hdf5_group was called on when
            this Distribution was saved
        
        Returns
        -------
        distribution : `UniformTriangulationDistribution`
            distribution created from the information in the given group
        """
        try:
            assert group.attrs['class'] == 'UniformTriangulationDistribution'
        except:
            raise TypeError("The given hdf5 file doesn't seem to contain a " +\
                "UniformTriangulationDistribution.")
        metadata = Distribution.load_metadata(group)
        points = get_hdf5_value(group['points'])
        return\
            UniformTriangulationDistribution(points=points, metadata=metadata)
    
    @property
    def gradient_computable(self):
        """
        Boolean describing whether the gradient of the given distribution has
        been implemented. If True,
        `UniformTriangulationDistribution.gradient_of_log_value` method can be
        called safely.
        """
        return True
    
    def gradient_of_log_value(self, point):
        """
        Computes the gradient (derivative) of the logarithm of the value
        of this `UniformTriangulationDistribution` at the given point.
        
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
        return np.zeros(self.numparams)
    
    @property
    def hessian_computable(self):
        """
        Boolean describing whether the hessian of the given distribution has
        been implemented. If True,
        `UniformTriangulationDistribution.hessian_of_log_value` method can be
        called safely.
        """
        return True
    
    def hessian_of_log_value(self, point):
        """
        Computes the hessian (second derivative) of the logarithm of the value
        of this `UniformTriangulationDistribution` at the given point.
        
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
        copied : `UniformTriangulationDistribution`
            a deep copy of this distribution, ignoring metadata.
        """
        return UniformTriangulationDistribution(\
            points=self.triangulation.points.copy())

