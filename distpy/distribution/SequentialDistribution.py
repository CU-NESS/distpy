"""
File: distpy/distribution/SequentialDistribution.py
Author: Keith Tauscher
Date: 12 Feb 2018

Description: File containing distribution which has a 1D form but manifests in
             the form of a sorted tuple.
"""
import numpy as np
from scipy.special import gammaln as log_gamma
from ..util import int_types, numerical_types, sequence_types
from .Distribution import Distribution

class SequentialDistribution(Distribution):
    """
    Class representing a distribution on parameters which must be in a specific
    order.
    """
    def __init__(self, shared_distribution, numpars=2, metadata=None):
        """
        Initializes a new SequentialDistribution.
        
        shared_distribution: the distribution from which values will be drawn
                             before they are sorted (must be univariate)
        numpars: number of parameters which this SequentialDistribution
                 describes
        """
        if isinstance(shared_distribution, Distribution):
            if shared_distribution.numparams == 1:
                self.shared_distribution = shared_distribution
            else:
                raise NotImplementedError("The shared_distribution " +\
                    "provided to a SequentialDistribution was multivariate " +\
                    "(I don't know how to deal with this!).")
        else:
            raise ValueError("The shared_distribution given to a " +\
                "SequentialDistribution was not recognizable as a " +\
                "distribution.")
        if (type(numpars) in numerical_types):
            if int(numpars) > 1:
                self._numparams = int(numpars)
            else:
                raise ValueError("A SequentialDistribution was initialized " +\
                    "with only one parameter. Is this really what you want?")
        else:
            raise ValueError("The type of the number of parameters given " +\
                "to a SequentialDistribution was not numerical.")
        self.metadata = metadata
    
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
                if np.all(point[1:] >= point[:-1]):
                    result = log_gamma(self.numparams + 1)
                    for ipar in range(self.numparams):
                        result +=\
                            self.shared_distribution.log_value(point[ipar])
                else:
                    return -np.inf
            else:
                raise ValueError("The length of the point provided to a " +\
                    "SequentialDistribution was not the same as the " +\
                    "SequentialDistribution's number of parameters")
        else:
            raise ValueError("The point given to a SequentialDistribution " +\
                "was not of a list type.")
        return result

    def to_string(self):
        """
        Finds and returns a string representation of this
        SequentialDistribution.
        """
        return "Sequential({!s})".format(self.shared_distribution.to_string())
    
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
            metadata_equal = self.metadata_equal(other)
            return all([numparams_equal, shared_distribution_equal,\
                metadata_equal])
        else:
            return False
    
    @property
    def is_discrete(self):
        """
        Property storing a boolean describing whether this distribution is
        discrete (True) or continuous (False).
        """
        return self.shared_distribution.is_discrete
    
    def fill_hdf5_group(self, group, save_metadata=True):
        """
        Fills the given hdf5 file group with data from this distribution. That
        data includes the class name, the number of parameters, and the shared
        distribution.
        
        group: hdf5 file group to fill
        save_metadata: if True, attempts to save metadata alongside
                                distribution and throws error if it fails
                       if False, metadata is ignored in saving process
        """
        group.attrs['class'] = 'SequentialDistribution'
        group.attrs['numparams'] = self.numparams
        subgroup = group.create_group('shared_distribution')
        self.shared_distribution.fill_hdf5_group(subgroup)
        if save_metadata:
            self.save_metadata(group)
    
    @staticmethod
    def load_from_hdf5_group(group, shared_distribution_class, *args,\
        **kwargs):
        """
        Loads a SequentialDistribution from the given hdf5 file group.
        
        group: the same hdf5 file group which fill_hdf5_group was called on
               when this Distribution was saved
        shared_distribution_class: the Distribution subclass which should be
                                   loaded from this Distribution
        args: positional arguments to pass on to load_from_hdf5_group method of
              shared_distribution_class
        kwargs: keyword arguments to pass on to load_from_hdf5_group method of
                shared_distribution_class
        
        returns: a SequentialDistribution object created from the information
                 in the given group
        """
        try:
            assert group.attrs['class'] == 'SequentialDistribution'
        except:
            raise TypeError("The given hdf5 file doesn't seem to contain a " +\
                "SequentialDistribution.")
        metadata = Distribution.load_metadata(group)
        numparams = group.attrs['numparams']
        shared_distribution = shared_distribution_class.load_from_hdf5_group(\
            group['shared_distribution'], *args, **kwargs)
        return SequentialDistribution(shared_distribution=shared_distribution,\
            numpars=numparams, metadata=metadata)
    
    @property
    def gradient_computable(self):
        """
        Property which stores whether the gradient of the given distribution
        has been implemented. Since it has been implemented, it returns True.
        """
        return self.shared_distribution.gradient_computable
    
    def gradient_of_log_value(self, point):
        """
        Computes the derivative of log_value(point) with respect to the
        parameter.
        
        point: vector of values at which to evaluate derivatives
        
        returns: returns single number representing derivative of log value
        """
        if np.all(point[1:] >= point[:-1]):
            answer = []
            for parameter in point:
                answer.append(\
                    self.shared_distribution.gradient_of_log_value(parameter))
            return np.array(answer)
        else:
            return np.zeros((self.numparams,))
    
    @property
    def hessian_computable(self):
        """
        Property which stores whether the hessian of the given distribution
        has been implemented. Since it has been implemented, it returns True.
        """
        return self.shared_distribution.hessian_computable
    
    def hessian_of_log_value(self, point):
        """
        Computes the second derivative of log_value(point) with respect to the
        parameter.
        
        point: vector of values at which to evaluate second derivatives
        
        returns: single number representing second derivative of log value
        """
        if np.all(point[1:] >= point[:-1]):
            answer = []
            for parameter in point:
                answer.append(\
                    self.shared_distribution.hessian_of_log_value(parameter))
            return np.diag(answer)
        else:
            return np.zeros((self.numparams,) * 2)

