"""
File: distpy/SequentialDistribution.py
Author: Keith Tauscher
Date: 6 Aug 2017

Description: File containing distribution which has a 1D form but manifests in
             the form of a sorted tuple.
"""
import numpy as np
from scipy.special import gammaln as log_gamma
from .TypeCategories import int_types, numerical_types, sequence_types
from .Distribution import Distribution
from .UniformDistribution import UniformDistribution


class SequentialDistribution(Distribution):
    """
    Class representing a distribution on parameters which must be in a specific
    order.
    """
    def __init__(self, shared_distribution=None, numpars=2):
        """
        Initializes a new SequentialDistribution.
        
        shared_distribution: the distribution from which values will be drawn
                             before they are sorted (must be univariate)
                             (defaults to Unif(0,1))
        numpars: number of parameters which this SequentialDistribution
                 describes
        """
        if shared_distribution is None:
            self.shared_distribution = UniformDistribution(0., 1.)
        elif isinstance(shared_distribution, Distribution):
            if shared_distribution.numparams == 1:
                self.shared_distribution = shared_distribution
            else:
                raise NotImplementedError("The shared_distribution " +\
                                          "provided to a " +\
                                          "SequentialDistribution was " +\
                                          "multivariate (I don't know how " +\
                                          "to deal with this!).")
        else:
            raise ValueError("The shared_distribution given to a " +\
                             "SequentialDistribution was not recognizable " +\
                             "as a distribution.")
        if (type(numpars) in numerical_types):
            if int(numpars) > 1:
                self._numparams = int(numpars)
            else:
                raise ValueError("A SequentialDistribution was initialized " +\
                                 "with only one parameter. Is this really " +\
                                 "what you want?")
        else:
            raise ValueError("The type of the number of parameters given " +\
                             "to a SequentialDistribution was not numerical.")

    @property
    def numparams(self):
        """
        Finds and returns the number of parameters which are described by this
        SequentialDistribution.
        """
        return self._numparams
    
    def draw(self, shape=None):
        """
        Draws values from shared_distribution and sorts them.
        
        shape: if None, returns single random variate
                        (scalar for univariate ; 1D array for multivariate)
               if int, n, returns n random variates
                          (1D array for univariate ; 2D array for multivariate)
               if tuple of n ints, returns that many random variates
                                   n-D array for univariate ;
                                   (n+1)-D array for multivariate
        
        returns numpy.ndarray of values (sorted by design)
        """
        none_shape = (shape is None)
        if none_shape:
            shape = (1,)
        elif type(shape) in int_types:
            shape = (shape,)
        unsorted = self.shared_distribution.draw(shape=shape+(self.numparams,))
        points = np.sort(np.array(unsorted))
        if none_shape:
            return points[0]
        else:
            return points

    def log_value(self, point):
        """
        Evaluates and returns the log_value at the given point. Point must be a
        numpy.ndarray (or other list-type) and if they are sorted, log_value
        returns -inf.
        """
        if type(point) in sequence_types:
            if len(point) == self.numparams:
                if all([point[ip] <= point[ip+1]\
                        for ip in range(len(point)-1)]):
                    result = log_gamma(self.numparams + 1)
                    for ipar in range(self.numparams):
                        result +=\
                            self.shared_distribution.log_value(point[ipar])
                else:
                    return -np.inf
            else:
                raise ValueError("The length of the point provided to a " +\
                                 "SequentialDistribution was not the same " +\
                                 "as the SequentialDistribution's number " +\
                                 "of parameters")
        else:
            raise ValueError("The point given to a SequentialDistribution " +\
                             "was not of a list type.")
        return result

    def to_string(self):
        """
        Finds and returns a string representation of this
        SequentialDistribution.
        """
        return "Sequential(%s)" % (self.shared_distribution.to_string(),)
    
    def __eq__(self, other):
        """
        Checks for equality of this distribution with other. Returns True if
        other is a SequentialDistribution with the same number of parameters
        and the same shared distribution and False otherwise.
        """
        if isinstance(other, SequentialDistribution):
            numparams_equal = (self.numparams == other.numparams)
            shared_distribution_equal =\
                (self.shared_distribution == other.shared_distribution)
            return numparams_equal and shared_distribution_equal
        else:
            return False
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data from this distribution. That
        data includes the class name, the number of parameters, and the shared
        distribution.
        
        group: hdf5 file group to fill
        """
        group.attrs['class'] = 'SequentialDistribution'
        group.attrs['numparams'] = self.numparams
        subgroup = group.create_group('shared_distribution')
        self.shared_distribution.fill_hdf5_group(subgroup)

