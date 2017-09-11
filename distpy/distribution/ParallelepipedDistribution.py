"""
File: distpy/ParallelepipedDistribution.py
Author: Keith Tauscher
Date: 6 Aug 2017

Description: File containing class represening a uniform distribution over an
             arbitrary parallelepiped (in arbitrary number of dimensions).
"""
import numpy as np
import numpy.random as rand
import numpy.linalg as lalg
from ..util import int_types, sequence_types
from .Distribution import Distribution

def _normed(vec):
    """
    Finds and returns a normalized version of the given vector.
    
    vec: 1D sequence of numbers
    """
    arrvec = np.array(vec)
    return (arrvec / lalg.norm(arrvec))

class ParallelepipedDistribution(Distribution):
    """
    Class representing a uniform distribution over a parallelepiped shaped
    region. The region is defined by constraints on linear combinations of the
    variables. See __init__ for more details.
    """
    def __init__(self, center, face_directions, distances, norm_dirs=True):
        """
        Initializes a new ParallelepipedDistribution.
        
        center the vector to the center of the region
        face_directions list of directions to the faces of the parallelepiped
        distances distances from center in given directions
        norm_dirs if True, then face_directions are normalized. This means that
                           the distances provided to this
                           method are "true distances"
                  if False, then face_directions are not normalized so the
                            region condition
                            dot(face_directions[i], from_center) < distances[i]
                            implies that distances are measured "in terms of
                            the combined quantity"
                            dot(face_directions[i], from_center)
        """
        if (type(center) in sequence_types):
            to_set = np.array(center)
            if (to_set.ndim == 1):
                self.center = to_set
                self._numparams = len(self.center)
            else:
                raise ValueError('The number of dimensions of the center ' +\
                                 'given to a ParallelepipedDistribution is ' +\
                                 ('not 1. It is %i-' % (to_set.ndim,)) +\
                                 'dimensional.')
        else:
            raise ValueError('A ParallelepipedDistribution was given with ' +\
                             'a center of an unrecognizable type.')
        if (type(face_directions) in sequence_types):
            to_set = np.matrix(face_directions)
            if (to_set.shape == ((self.numparams,) * 2)):
                if norm_dirs:
                    self.face_directions =\
                        np.matrix([_normed(face_directions[i])\
                                   for i in range(self.numparams)])
                else:
                    self.face_directions =\
                        np.matrix([face_directions[i]\
                                   for i in range(self.numparams)])
            else:
                raise ValueError('The shape of the face directions in ' +\
                                 'matrix form was not the expected value, ' +\
                                 'which is (self.numparams, self.numparams).')
        else:
            raise ValueError('A ParallelepipedDistribution was given ' +\
                             'face_directions of an unrecognizable type.')
        if (type(distances) in sequence_types):
            arrdists = np.array(distances)
            if (arrdists.ndim == 1) and (len(arrdists) == self.numparams):
                self.distances = arrdists
            else:
                raise ValueError('distances given to ' +\
                                 'ParallelepipedDistribution are either of ' +\
                                 'the wrong dimension or the wrong length.')
        self.inv_face_directions = lalg.inv(self.face_directions)

    @property
    def numparams(self):
        """
        Finds and returns the number of parameters to which this
        ParallelepipedDistribution applies.
        """
        if not hasattr(self, '_numparams'):
            raise AttributeError("For some reason, I don't know how many " +\
                                 "params this ParallelepipedDistribution " +\
                                 "describes!")
        return self._numparams

    @property
    def matrix(self):
        """
        Finds the matrix whose rows are vectors pointing from the vertex to
        all adjacent vertices.
        
        returns the matrix which has the directions from the vertex as its rows
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
        Finds and returns the vertex which satisfies:
        
        (vec(v)-vec(c)) dot face_directions[i] == distances[i]   for all i
        
        where vec(v) is the vector to the vertex and vec(c) is the vector to
        the center
        """
        if not hasattr(self, '_vertex'):
            from_cent = self.inv_face_directions * np.matrix(self.distances).T
            from_cent = np.array(from_cent).squeeze()
            self._vertex = self.center + from_cent
        return self._vertex

    @property
    def area(self):
        """
        Finds the "area" (more like hypervolume in the general case) of the
        parallelepiped-shaped region described by this
        ParallelepipedDistribution.
        """
        if not hasattr(self, '_area'):
            self._area = np.abs(lalg.det(self.matrix))
        return self._area

    def draw(self, shape=None):
        """
        Draws a value from the parallelepiped this object describes (uniform
        distribution over support).
        
        shape: if None, returns single random variate
                        (scalar for univariate ; 1D array for multivariate)
               if int, n, returns n random variates
                          (1D array for univariate ; 2D array for multivariate)
               if tuple of n ints, returns that many random variates
                                   n-D array for univariate ;
                                   (n+1)-D array for multivariate
        
        returns random draw in form of numpy.ndarray
        """
        none_shape = (shape is None)
        if none_shape:
            shape = (1,)
        elif type(shape) in int_types:
            shape = (shape,)
        transformed_point = rand.rand(*(shape + (self.numparams,)))
        points = self.vertex + np.dot(transformed_point, self.matrix.A)
        if none_shape:
            return points[0]
        else:
            return points

    def log_value(self, point):
        """
        Computes the log of the value of this distribution at the given point.
        
        point the point at which to evaluate the log value; a numpy.ndarray
        
        returns: log of the value of the distribution at the given point
        """
        if self._in_region(point):
            return -np.log(self.area)
        return -np.inf

    def to_string(self):
        """
        Finds and returns a string representation of this
        ParallelepipedDistribution.
        """
        return "Parallelepiped(%s, %s, %s)" %\
            (self.center, self.face_directions, self.distance)

    def _in_region(self, point):
        #
        # Finds if the given point is in the region defined by this
        # ParallelepipedDistribution.
        #
        # point the point to test for inclusion
        #
        # returns True if point in region, False otherwise
        #
        if type(point) not in sequence_types:
            raise ValueError('point given to log_value ' +\
                             'was not of an arraylike type.')
        arrpoint = np.array(point)
        if (arrpoint.ndim != 1) or (len(arrpoint) != self.numparams):
            raise ValueError('The point given is either of the ' +\
                             'wrong direction or the wrong length.')
        from_center = arrpoint - self.center
        return_val = True
        for i in range(self.numparams):
            dotp = np.dot(from_center, self.face_directions.A[i,:])
            return_val =\
                (return_val and (np.abs(dotp) <= np.abs(self.distances[i])))
        return return_val
    
    def __eq__(self, other):
        """
        Checks for equality of this distribution with other. Returns True if
        other is a ParallelepipedDistribution with the same center,
        face_directions, and distances (to a dynamic range of 10^-9) and False
        otherwise.
        """
        if isinstance(other, ParallelepipedDistribution):
            center_close =\
                np.allclose(self.center, other.center, rtol=1e-9, atol=0)
            face_directions_close = np.allclose(self.face_directions.A,\
                other.face_directions.A, rtol=1e-9, atol=0)
            distances_close =\
                np.allclose(self.distances, other.distances, rtol=1e-9, atol=0)
            return center_close and face_directions_close and distances_close
        else:
            return False
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data from this distribution. The
        class name of the distribution is saved along with the center,
        face_directions, and distances.
        
        group: hdf5 file group to fill
        """
        group.attrs['class'] = 'ParallelepipedDistribution'
        group.create_dataset('center', data=self.center)
        group.create_dataset('face_directions', data=self.face_directions)
        group.create_dataset('distances', data=self.distances)
