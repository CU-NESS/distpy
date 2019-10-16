"""
File: distpy/distribution/UniformTriangulationDistribution.py
Author: Keith Tauscher
Date: Oct 15 2019

Description: File containing a class representing a uniform distribution over a
             set of simplices (usually a Delaunay triangulation).
"""
import numpy as np
import numpy.linalg as la
import numpy.random as rand
from scipy.spatial import Delaunay
from ..util import int_types, create_hdf5_dataset, get_hdf5_value
from .Distribution import Distribution

class UniformTriangulationDistribution(Distribution):
    """
    Class representing a uniform distribution over the convex hull of given
    points.
    """
    def __init__(self, triangulation=None, points=None, metadata=None):
        """
        Creates a new UniformTriangulationDistribution with the given
        triangulation.
        
        triangulation: an object which implements points like the
                       scipy.spatial.Delaunay class. if None, points can be
                       given instead.
        points: only used if triangulation is None, points with which to
                compute triangulation
        """
        self.triangulation = (triangulation, points)
        self.metadata = metadata
    
    @property
    def triangulation(self):
        """
        Property storing the triangulation at the heart of this distribution.
        """
        if not hasattr(self, '_triangulation'):
            raise AttributeError("triangulation was referenced before it " +\
                "was set.")
        return self._triangulation
    
    @triangulation.setter
    def triangulation(self, value):
        """
        Setter for the triangulation at the heart of this distribution.
        
        value: tuple of form (triangulation, points) where one and only one of
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
        Property storing the number of parameters associated with the given
        triangulation.
        """
        if not hasattr(self, '_numparams'):
            self._numparams = self.triangulation.points.shape[1]
        return self._numparams
    
    @property
    def mean(self):
        """
        Property storing the mean of this distribution.
        """
        if not hasattr(self, '_mean'):
            raise AttributeError("mean is not implemented for the " +\
                "UniformTriangulationDistribution class.")
        return self._mean
    
    @property
    def variance(self):
        """
        Property storing the covariance of this distribution.
        """
        if not hasattr(self, '_variance'):
            raise AttributeError("variance is not implemented for the " +\
                "UniformTriangulationDistribution class.")
        return self._variance
    
    @property
    def nsimplices(self):
        """
        Property storing the number of simplices in the triangulation at the
        heart of this distribution.
        """
        if not hasattr(self, '_nsimplices'):
            self._nsimplices = len(self.triangulation.simplices)
        return self._nsimplices

    @property
    def numparams_factorial(self):
        """
        Property storing the factorial of the number of parameters.
        """
        if not hasattr(self, '_numparams_factorial'):
            self._numparams_factorial = np.prod(np.arange(self.numparams) + 1)
        return self._numparams_factorial
    
    @property
    def simplex_volumes(self):
        """
        Property storing the volumes of the simplices in the Triangulation at
        the heart of this Distribution.
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
        Property storing the volume of the convex hull of the simplices in the
        triangulation.
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
        """
        temp_space = random.uniform(size=self.numparams)
        return np.diff(np.sort(np.concatenate([[0], temp_space, [1]])))
    
    def draw_from_simplex(self, isimplex, random=rand):
        """
        Draws a point uniformly from the simplex associated with the given
        index.
        
        isimplex: index associated with the simplex to sample
        
        returns: a single point from the given simplex
        """
        vertices =\
            self.triangulation.points[self.triangulation.simplices[isimplex],:]
        return np.dot(self.draw_coefficients(random=random), vertices)

    def draw(self, shape=None, random=rand):
        """
        Draws and returns a value from this distribution using numpy.random.
        
        shape: if None, returns single random variate
                        (scalar for univariate ; 1D array for multivariate)
               if int, n, returns n random variates
                          (1D array for univariate ; 2D array for multivariate)
               if tuple of n ints, returns that many random variates
                                   n-D array for univariate ;
                                   (n+1)-D array for multivariate
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
        Property storing the log of the value of this distribution inside the
        convex hull of the given points. If it is outside the convex hull, then
        it returns instead -np.inf as the log value.
        """
        if not hasattr(self, '_constant_log_value'):
            self._constant_log_value = -np.log(self.total_volume)
        return self._constant_log_value

    def log_value(self, point):
        """
        Evaluates and returns the log of the value of this distribution when
        the variable is value.
        
        point: numerical value of the variable
        """
        if self.triangulation.find_simplex(point) == -1:
            return -np.inf
        else:
            return self.constant_log_value
    
    def to_string(self):
        """
        Finds and returns a string representation of this distribution.
        """
        return "UniformTriangulationDistribution"
    
    def __eq__(self, other):
        """
        Checks for equality of this distribution with other. Returns True if
        other is a UniformDistribution with the same high and low (down to 1e-9
        level) and False otherwise.
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
        Property storing the minimum allowable value(s) in this distribution.
        """
        # TODO make this actual minimum coordinates of convex hull
        return [None] * self.numparams
    
    @property
    def maximum(self):
        """
        Property storing the maximum allowable value(s) in this distribution.
        """
        # TODO make this actual maximum coordinates of conves hull
        return [None] * self.numparams
    
    @property
    def is_discrete(self):
        """
        Property storing a boolean describing whether this distribution is
        discrete (True) or continuous (False).
        """
        return False
    
    def fill_hdf5_group(self, group, save_metadata=True):
        """
        Fills the given hdf5 file group with data from this distribution. All
        that needs to be saved is the class name and high and low values.
        
        group: hdf5 file group to fill
        save_metadata: if True, attempts to save metadata alongside
                                distribution and throws error if it fails
                       if False, metadata is ignored in saving process
        """
        group.attrs['class'] = 'UniformTriangulationDistribution'
        create_hdf5_dataset(group, 'points', data=self.triangulation.points)
        if save_metadata:
            self.save_metadata(group)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a UniformTriangulationDistribution from the given hdf5 file
        group.
        
        group: the same hdf5 file group which fill_hdf5_group was called on
               when this Distribution was saved
        
        returns: a UniformTriangulationDistribution object created from the
                 information in the given group
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
        Property which stores whether the gradient of the given distribution
        has been implemented. Since it has been implemented, it returns True.
        """
        return True
    
    def gradient_of_log_value(self, point):
        """
        Computes the derivative of log_value(point) with respect to the
        parameter.
        
        point: numpy.ndarray point at which to evaluate the derivative
        
        returns: returns numpy.ndarray of same shape as point representing
                 derivative of log value
        """
        return np.zeros(self.numparams)
    
    @property
    def hessian_computable(self):
        """
        Property which stores whether the hessian of the given distribution
        has been implemented. Since it has been implemented, it returns True.
        """
        return True
    
    def hessian_of_log_value(self, point):
        """
        Computes the second derivative of log_value(point) with respect to the
        parameter.
        
        point: numpy.ndarray point at which to evaluate the hessian
        
        returns: single number representing second derivative of log value
        """
        return np.zeros((self.numparams,) * 2)
    
    def copy(self):
        """
        Returns a deep copy of this Distribution. This function ignores
        metadata.
        """
        return UniformTriangulationDistribution(\
            points=self.triangulation.points.copy())

