"""
Module containing class representing a distribution that is uniform over a
parallelepiped in an arbitrary number of dimensions.

**File**: $DISTPY/distpy/distribution/ParallelepipedDistribution.py  
**Author**: Keith Tauscher  
**Date**: 31 May 2021
"""
import numpy as np
import numpy.random as rand
import numpy.linalg as lalg
from ..util import int_types, sequence_types, create_hdf5_dataset,\
    get_hdf5_value
from .Distribution import Distribution

def _normed(vec):
    """
    Finds and returns a normalized version of the given vector.
    
    Parameters
    ----------
    vec : sequence
        vector to norm
    
    Returns
    -------
    normed : `numpy.ndarray`
        if `vec` is \\(\\boldsymbol{x}\\), then `normed` is
        \\(\\boldsymbol{x}/\\sqrt{\\boldsymbol{x}\\cdot\\boldsymbol{x}}\\)
    """
    arrvec = np.array(vec)
    return (arrvec / np.sqrt(np.vdot(arrvec, arrvec)))

class ParallelepipedDistribution(Distribution):
    """
    Class representing a distribution that is uniform over a parallelepiped in
    an arbitrary number of dimensions.
    """
    def __init__(self, center, face_directions, distances, metadata=None):
        """
        Initializes a new `ParallelepipedDistribution` with the given parameter
        values.
        
        Parameters
        ----------
        center : `numpy.ndarray`
            array describing vector pointing to center of parallelepiped
        face_directions : sequence
            sequence of arrays giving unit vectors from center of
            parallelepiped to its faces
        distances : `numpy.ndarray`
            array of distances to each face from the center
        metadata : number or str or dict or `distpy.util.Savable.Savable`
            data to store alongside this distribution.
        """
        self.center = center
        self.face_directions = face_directions
        self.distances = distances
        self.metadata = metadata
    
    @property
    def center(self):
        """
        The center point of the parallelepiped.
        """
        if not hasattr(self, '_center'):
            raise AttributeError("center was referenced before it was set.")
        return self._center
    
    @center.setter
    def center(self, value):
        """
        Setter for `ParallelepipedDistribution.center`.
        
        Parameters
        ----------
        value : numpy.ndarray
            1D numpy.ndarray of length `ParallelepipedDistribution.numparams`
        """
        if (type(value) in sequence_types):
            value = np.array(value)
            if (value.ndim == 1):
                self._center = value
            else:
                raise ValueError(('The number of dimensions of the center ' +\
                    'given to a ParallelepipedDistribution is not 1. It ' +\
                    'is {}-dimensional.').format(value.ndim))
        else:
            raise ValueError('A ParallelepipedDistribution was given with ' +\
                'a center of an unrecognizable type.')
    
    @property
    def face_directions(self):
        """
        A matrix encoding the directions to each face of the parallelepiped.
        """
        if not hasattr(self, '_face_directions'):
            raise AttributeError("face_directions was referenced before it " +\
                "was set.")
        return self._face_directions
    
    @face_directions.setter
    def face_directions(self, value):
        """
        Setter for `ParallelepipedDistribution.face_directions`.
        
        Parameters
        ----------
        value : sequence
            list of directions to the faces of the parallelepiped. These will
            be normalized
        """
        if (type(value) in sequence_types):
            value = np.array(value)
            if (value.shape == ((self.numparams,) * 2)):
                self._face_directions = [_normed(value[i])\
                    for i in range(self.numparams)]
                self._face_directions = np.matrix(self._face_directions)
            else:
                raise ValueError("The shape of the face_directions in " +\
                    "matrix form was not the expected value, which is " +\
                    "(self.numparams, self.numparams).")
        else:
            raise ValueError("A ParallelepipedDistribution was given " +\
                "face_directions of an unrecognizable type.")
    
    @property
    def distances(self):
        """
        The distances to each face of the parallelepiped.
        """
        if not hasattr(self, '_distances'):
            raise AttributeError("distances was referenced before it was set.")
        return self._distances
    
    @distances.setter
    def distances(self, value):
        """
        Setter for `ParallelepipedDistribution.distances`.
        
        Parameter
        ---------
        value : numpy.ndarray
            1D array of positive numbers with the same shape as
            `ParallelepipedDistribution.center`
        """
        if (type(value) in sequence_types):
            value = np.array(value)
            if value.shape == (self.numparams,):
                if np.all(value > 0):
                    self._distances = value
                else:
                    raise ValueError("Not all distances were positive.")
            else:
                raise ValueError("distances given to " +\
                    "ParallelepipedDistribution have the wrong shape.")
        else:
            raise TypeError("distances was set to a non-sequence.")
    
    @property
    def inv_face_directions(self):
        """
        The inverse of the matrix describing the directions to the faces of the
        parallelepiped.
        """
        if not hasattr(self, '_inv_face_directions'):
            self._inv_face_directions = lalg.inv(self.face_directions)
        return self._inv_face_directions

    @property
    def numparams(self):
        """
        The number of parameters of this `ParallelepipedDistribution`.
        """
        if not hasattr(self, '_numparams'):
            self._numparams = len(self.center)
        return self._numparams
    
    @property
    def mean(self):
        """
        The mean of this `ParallelepipedDistribution`, which is the center of
        the parallelepiped.
        """
        if not hasattr(self, '_mean'):
            self._mean = self.center
        return self._mean
    
    @property
    def variance(self):
        """
        The variance of the `ParallelepipedDistribution` class is not
        implemented.
        """
        if not hasattr(self, '_variance'):
            raise AttributeError("variance is not implemented for the " +\
                "ParallelepipedDistribution class.")
        return self._variance

    @property
    def matrix(self):
        """
        The matrix whose rows are vectors pointing from the vertex to all
        adjacent vertices.
        """
        if not hasattr(self, '_matrix'):
            def row(index):
                #
                # Finds the index'th row of the matrix. Essentially, this is
                # the vector from the vertex to the index'th adjacent vertex.
                #
                mod_dists = self.distances.copy()
                mod_dists[index] = (mod_dists[index] * (-1))
                from_cent = self.inv_face_directions * np.matrix(mod_dists).T
                from_cent = np.array(from_cent).squeeze()
                return self.center + from_cent - self.vertex
            self._matrix = np.matrix([row(i) for i in range(self.numparams)])
        return self._matrix

    @property
    def vertex(self):
        """
        The vertex which satisfies
        \\((\\boldsymbol{v}-\\boldsymbol{c})\\cdot\\boldsymbol{\\hat{n}}_i=\
        d_i\\) for all \\(k\\), where \\(\\boldsymbol{v}\\) is the vertex,
        \\(\\boldsymbol{c}\\) is the center, \\(\\boldsymbol{\\hat{n}}_k\\) is
        the \\(k^{\\text{th}}\\) normalized face direction to the, and
        \\(d_k\\) is the distance to the \\(k^{\\text{th}}\\) face.
        """
        if not hasattr(self, '_vertex'):
            from_cent = self.inv_face_directions * np.matrix(self.distances).T
            from_cent = np.array(from_cent).squeeze()
            self._vertex = self.center + from_cent
        return self._vertex

    @property
    def area(self):
        """
        The "area" (more like hypervolume in the general case) of the
        parallelepiped-shaped region described by this
        `ParallelepipedDistribution`.
        """
        if not hasattr(self, '_area'):
            self._area = np.abs(lalg.det(self.matrix))
        return self._area

    def draw(self, shape=None, random=rand):
        """
        Draws point(s) from this `ParallelepipedDistribution`. Below, `p` is
        `ParallelepipedDistribution.numparams`.
        
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
        transformed_point = random.rand(*(shape + (self.numparams,)))
        points = self.vertex + np.dot(transformed_point, self.matrix.A)
        if none_shape:
            return points[0]
        else:
            return points

    def log_value(self, point):
        """
        Computes the logarithm of the value of this
        `ParallelepipedDistribution` at the given point.
        
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
        if self._in_region(point):
            return -np.log(self.area)
        return -np.inf

    def to_string(self):
        """
        Finds and returns a string version of this `ParallelepipedDistribution`
        of the form `"Parallelepiped(center, face_directions, distance)"`.
        """
        return "Parallelepiped({0!s}, {1!s}, {2!s})".format(self.center,\
            self.face_directions, self.distances)

    def _in_region(self, point):
        """
        Finds if the given point is in the region defined by this
        ParallelepipedDistribution.
        
        Parameters
        ----------
        point : numpy.ndarray
            the point to test for inclusion
        
        Returns
        -------
        containment : bool
            True if point in region, False otherwise
        """
        if type(point) not in sequence_types:
            raise ValueError('point given to log_value was not of an ' +\
                'array-like type.')
        arrpoint = np.array(point)
        if (arrpoint.ndim != 1) or (len(arrpoint) != self.numparams):
            raise ValueError('The point given is either of the wrong ' +\
                'direction or the wrong length.')
        from_center = arrpoint - self.center
        return_val = True
        for i in range(self.numparams):
            dotp = np.dot(from_center, self.face_directions.A[i,:])
            return_val =\
                (return_val and (np.abs(dotp) <= np.abs(self.distances[i])))
        return return_val
    
    def __eq__(self, other):
        """
        Checks for equality of this `ParallelepipedDistribution` with `other`.
        
        Parameters
        ----------
        other : object
            object to check for equality
        
        Returns
        -------
        result : bool
            True if and only if `other` is a `ParallelepipedDistribution` with
            the same `ParallelepipedDistribution.center`,
            `ParallelepipedDistribution.face_directions`, and
            `ParallelepipedDistribution.distances`
        """
        if isinstance(other, ParallelepipedDistribution):
            tol_kwargs = {'rtol': 1e-9, 'atol': 0.}
            center_close = np.allclose(self.center, other.center, **tol_kwargs)
            face_directions_close = np.allclose(self.face_directions.A,\
                other.face_directions.A, **tol_kwargs)
            distances_close =\
                np.allclose(self.distances, other.distances, **tol_kwargs)
            metadata_equal = self.metadata_equal(other)
            return all([center_close, face_directions_close, distances_close,\
                metadata_equal])
        else:
            return False
    
    @property
    def can_give_confidence_intervals(self):
        """
        Multivariate distributions do not support confidence intervals.
        """
        return False
    
    @property
    def minimum(self):
        """
        The minimum allowable value(s) in this distribution.
        """
        return self.center - np.abs(self.vertex - self.center)
    
    @property
    def maximum(self):
        """
        The maximum allowable value(s) in this distribution.
        """
        return self.center + np.abs(self.vertex - self.center)
    
    @property
    def is_discrete(self):
        """
        Boolean describing whether this distribution is discrete (True) or
        continuous (False).
        """
        return False
    
    def fill_hdf5_group(self, group, center_link=None,\
        face_directions_link=None, distances_link=None, save_metadata=True):
        """
        Fills the given hdf5 file group with data about this
        `ParallelepipedDistribution` so that it can be loaded later.
        
        Parameters
        ----------
        group : h5py.Group
            hdf5 file group to fill
        center_link : str or h5py.Dataset or None
            link to mean in hdf5 file, if it exists
        face_directions_link : str or h5py.Dataset or None
            link to face_directions in hdf5 file, if it exists
        distances_link : str or h5py.Dataset or None
            link to distances to faces in hdf5 file, if it exists
        save_metadata : bool
            - if True, attempts to save metadata alongside distribution and
            throws error if it fails
            - if False, metadata is ignored in saving process
        """
        group.attrs['class'] = 'ParallelepipedDistribution'
        create_hdf5_dataset(group, 'center', data=self.center,\
            link=center_link)
        create_hdf5_dataset(group, 'face_directions',\
            data=self.face_directions, link=face_directions_link)
        create_hdf5_dataset(group, 'distances', data=self.distances,\
            link=distances_link)
        if save_metadata:
            self.save_metadata(group)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a `ParallelepipedDistribution` from the given hdf5 file group.
        
        Parameters
        ----------
        group : h5py.Group
            the same hdf5 file group which fill_hdf5_group was called on when
            this Distribution was saved
        
        Returns
        -------
        distribution : `ParallelepipedDistribution`
            distribution created from the information in the given group
        """
        try:
            assert group.attrs['class'] == 'ParallelepipedDistribution'
        except:
            raise TypeError("The given hdf5 file doesn't seem to contain a " +\
                "ParallelepipedDistribution.")
        metadata = Distribution.load_metadata(group)
        center = get_hdf5_value(group['center'])
        face_directions = get_hdf5_value(group['face_directions'])
        distances = get_hdf5_value(group['distances'])
        return ParallelepipedDistribution(center, face_directions, distances,\
            metadata=metadata)
    
    @property
    def gradient_computable(self):
        """
        Boolean describing whether the gradient of the given distribution has
        been implemented. If True,
        `ParallelepipedDistribution.gradient_of_log_value` method can be called
        safely.
        """
        return True
    
    def gradient_of_log_value(self, point):
        """
        Computes the gradient (derivative) of the logarithm of the value of
        this `ParallelepipedDistribution` at the given point.
        
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
        `ParallelepipedDistribution.hessian_of_log_value` method can be called
        safely.
        """
        return True
    
    def hessian_of_log_value(self, point):
        """
        Computes the hessian (second derivative) of the logarithm of the value
        of this `ParallelepipedDistribution` at the given point.
        
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
        copied : `ParallelepipedDistribution`
            a deep copy of this distribution, ignoring metadata.
        """
        return ParallelepipedDistribution(self.center.A[0].copy(),\
            self.face_directions.A.copy(), self.distances.copy())

