"""
File: distpy/distribution/SparseGaussianDistribution.py
Author: Keith Tauscher
Date: May 13 2021

Description: File containing class representing Gaussian distribution with a
             block diagonal covariance matrix.
"""
from __future__ import division
import numpy as np
import numpy.random as rand
import numpy.linalg as npla
import scipy.linalg as scila
from scipy import sparse
from ..util import numerical_types, int_types, sequence_types,\
    create_hdf5_dataset, get_hdf5_value
from .Distribution import Distribution
from .GaussianDistribution import GaussianDistribution

natural_log_two_pi = np.log(2 * np.pi)

class SparseGaussianDistribution(Distribution):
    """
    Class representing Gaussian distribution with a block diagonal covariance
    matrix.
    """
    def __init__(self, mean, covariance_list, metadata=None):
        """
        Initializes either a univariate or a multivariate
        SparseGaussianDistribution.
        
        mean: a 1D array of numbers
        covariance_list: a list of 2D arrays that are the blocks on the
                         diagonal of the covariance matrix. The sum of sizes of
                         the blocks should be the same as the length of mean
        """
        self.mean = mean
        self.covariance_list = covariance_list
        self.metadata = metadata
    
    @staticmethod
    def combine(*distributions):
        """
        Combines many SparseGaussianDistribution objects into one by
        concatenating their means and covariance matrices.
        
        *distributions: a sequence of SparseGaussianDistribution and/or
                        GaussianDistribution objects to combine
        
        returns: a single SparseGaussianDistribution object 
        """
        if all([(type(distribution) in\
            [GaussianDistribution, SparseGaussianDistribution])\
            for distribution in distributions]):
            (mean, covariance_list) = ([], [])
            for distribution in distributions:
                if isinstance(distribution, GaussianDistribution):
                    mean.append(distribution.mean.A[0])
                    covariance_list.append(distribution.covariance.A)
                else:
                    mean.append(distribution.mean)
                    covariance_list.extend(distribution.covariance_list)
            mean = np.concatenate(mean)
            return SparseGaussianDistribution(mean, covariance_list)
        else:
            raise TypeError("At least one of the distributions given to " +\
                "the SparseGaussianDistribution class' combine function " +\
                "was neither a GaussianDistribution nor a " +\
                "SparseGaussianDistribution.")
    
    @property
    def mean(self):
        """
        Property storing the mean of this SparseGaussianDistribution in array
        form.
        """
        if not hasattr(self, '_mean'):
            raise AttributeError("mean was referenced before it was set.")
        return self._mean
    
    @mean.setter
    def mean(self, value):
        """
        Setter for the mean of this distribution
        
        value: a 1D numpy.ndarray of length numparams
        """
        if type(value) in numerical_types:
            value = [value]
        if type(value) in sequence_types:
            value = np.array(value)
            if value.ndim != 1:
                raise ValueError("The mean of a SparseGaussianDistribution " +\
                    "was not 1 dimensional.")
            elif value.size == 0:
                raise ValueError("The mean of a SparseGaussianDistribution " +\
                    "was set to something like an empty array.")
            elif value.size == 1:
                raise ValueError("The mean of a SparseGaussianDistribution " +\
                    "was given as a single number or sequence of length 1. " +\
                    "This would imply that this distribution should be " +\
                    "univariate. However, the SparseGaussianDistribution " +\
                    "class is meant to for large numbers of parameters. " +\
                    "The standard GaussianDistribution class should be " +\
                    "used for a univariate Gaussian distribution.")
            else:
                self._mean = value
        else:
            raise ValueError("The mean of a GaussianDistribution is not of " +\
                "a recognizable type.")

    @property
    def numparams(self):
        """
        Finds and returns the number of parameters which this
        GaussianDistribution describes (same as dimension of mean and
        covariance).
        """
        if not hasattr(self, '_numparams'):
            self._numparams = len(self.mean)
        return self._numparams
    
    @property
    def covariance_list(self):
        """
        Property storing the list of covariance blocks.
        """
        if not hasattr(self, '_covariance_list'):
            raise AttributeError("covariance_list was referenced before it " +\
                "was set.")
        return self._covariance_list
    
    @covariance_list.setter
    def covariance_list(self, value):
        """
        Setter for the list of covariance blocks.
        
        value: a list of 2D arrays that are the blocks on the diagonal of the
               covariance matrix. The sum of sizes of the blocks should be the
               same as the length of mean
        """
        if type(value) in sequence_types:
            new_value = []
            for element in value:
                if type(element) in numerical_types:
                    new_element = np.array([[element]])
                elif type(element) in sequence_types:
                    new_element = np.array(element)
                    new_element = (new_element + new_element.T) / 2
                else:
                    raise TypeError("At least one block of the covariance " +\
                        "was set to neither a number nor a sequence.")
                new_value.append(new_element)
            if all([((element.ndim == 2) and\
                (element.shape[0] == element.shape[1]))\
                for element in new_value]):
                if sum([element.shape[0] for element in new_value]) ==\
                    self.numparams:
                    self._covariance_list = new_value
                else:
                    raise ValueError("The sums of the sizes of the blocks " +\
                        "in covariance_list was not the same as the length " +\
                        "of the mean vector.")
            else:
                raise ValueError("At least one block of the " +\
                    "covariance_list was not square.")
        else:
            raise TypeError("covariance_list was set to a non-sequence.")
    
    @property
    def covariance(self):
        """
        Property storing the covariance matrix of this Gaussian as a
        scipy.sparse.spmatrix.
        """
        if not hasattr(self, '_covariance'):
            self._covariance = sparse.block_diag(self.covariance_list)
        return self._covariance
    
    @property
    def mean_list(self):
        """
        Property storing the means corresponding to each covariance block.
        """
        if not hasattr(self, '_mean_list'):
            current_index = 0
            self._mean_list = []
            for block in self.covariance_list:
                block_size = len(block)
                self._mean_list.append(self.mean[current:current+block_size])
                current = current + block_size
        return self._mean_list
    
    @property
    def log_determinant_covariance(self):
        """
        Property storing the natural logarithm of the determinant of the
        covariance matrix.
        """
        if not hasattr(self, '_log_determinant_covariance'):
            self._log_determinant_covariance =\
                sum([npla.slogdet(block)[1] for block in self.covariance_list])
        return self._log_determinant_covariance
    
    def _make_inverse_and_square_root_covariance_lists(self):
        """
        Finds the inverse, square root, and inverse square root of the
        covariance blocks and stores them in internal properties, allowing
        inverse_covariance, square_root_covariance, and
        inverse_square_root_covariance properties to be referenced.
        """
        self._square_root_covariance_list = []
        self._inverse_covariance_list = []
        self._inverse_square_root_covariance_list = []
        for block in self.covariance_list:
            (eigenvalues, eigenvectors) = npla.eigh(block)
            if np.any(eigenvalues <= 0):
                raise ValueError(("Something went wrong, causing the " +\
                    "square root of a block of the the covariance matrix " +\
                    "of this SparseGaussianDistribution to have at least " +\
                    "one complex element. The eigenvalues of the " +\
                    "covariance matrix block are {!s}.").format(eigenvalues))
            eigenvalues = np.sqrt(eigenvalues)
            self._square_root_covariance_list.append(np.dot(\
                eigenvectors * np.sqrt(eigenvalues)[None,:], eigenvectors.T))
            self._inverse_covariance_list.append(\
                np.dot(eigenvectors / eigenvalues[None,:], eigenvectors.T))
            self._inverse_square_root_covariance_list.append(np.dot(\
                eigenvectors / np.sqrt(eigenvalues)[None,:], eigenvectors.T))
    
    @property
    def inverse_covariance_list(self):
        """
        Property storing the list of inverses of the covariance blocks.
        """
        if not hasattr(self, '_inverse_covariance_list'):
            self._make_inverse_and_square_root_covariance_lists()
        return self._inverse_covariance_list
    
    @property
    def inverse_covariance(self):
        """
        Property storing the inverse of the covariance as a
        scipy.sparse.spmatrix.
        """
        if not hasattr(self, '_inverse_covariance'):
            self._inverse_covariance =\
                sparse.block_diag(self.inverse_covariance_list)
        return self._inverse_covariance
    
    @property
    def square_root_covariance_list(self):
        """
        Property storing the list of square roots of the covariance blocks.
        """
        if not hasattr(self, '_square_root_covariance_list'):
            self._make_inverse_and_square_root_covariance_lists()
        return self._square_root_covariance_list
    
    @property
    def square_root_covariance(self):
        """
        Property storing the square root of the covariance matrix as a
        scipy.sparse.spmatrix.
        """
        if not hasattr(self, '_square_root_covariance'):
            self._square_root_covariance =\
                sparse.block_diag(self.square_root_covariance_list)
        return self._square_root_covariance
    
    @property
    def inverse_square_root_covariance_list(self):
        """
        Property storing the list of inverse square roots of the covariance
        blocks.
        """
        if not hasattr(self, '_inverse_square_root_covariance_list'):
            self._make_inverse_and_square_root_covariance_lists()
        return self._inverse_square_root_covariance_list
    
    @property
    def inverse_square_root_covariance(self):
        """
        Property storing the inverse square root of the covariance matrix as a
        scipy.sparse.spmatrix.
        """
        if not hasattr(self, '_inverse_square_root_covariance'):
            self._inverse_square_root_covariance =\
                sparse.block_diag(self.inverse_square_root_covariance_list)
        return self._inverse_square_root_covariance
    
    def weight(self, array, axis=0):
        """
        Weights the given array by the inverse square root of the covariance
        matrix of this distribution.
        
        array: the array to weight, can be any number of dimensions as long as
               the specified one has length self.numparams
        axis: index of the axis corresponding to the parameters
        
        returns: numpy.ndarray of same shape as array corresponding to
                 \(C^{-1/2} A\) where A is array shaped so that the matrix
                 multiplication makes sense.
        """
        axis = axis % array.ndim
        if array.ndim == 1:
            return self.inverse_square_root_covariance.dot(array)
        elif array.ndim == 2:
            if axis == 0:
                return self.inverse_square_root_covariance.dot(array)
            else:
                return self.inverse_square_root_covariance.dot(array.T).T
        else:
            before_shape = array.shape[:axis]
            after_shape = array.shape[(axis+1):]
            if axis != 0:
                weighted_array = np.rollaxis(array, axis, start=0)
            weighted_array = np.reshape(weighted_array, (self.numparams, -1))
            weighted_array =\
                self.inverse_square_root_covariance.dot(weighted_array)
            weighted_array = np.reshape(weighted_array,\
                (self.numparams,) + before_shape + after_shape)
            if axis != 0:
                weighted_array = np.rollaxis(weighted_array, 0, start=axis+1)
            return weighted_array
    
    @property
    def variance(self):
        """
        Property storing the covariance of this distribution.
        """
        if not hasattr(self, '_variance'):
            self._variance = self.covariance.toarray()
        return self._variance
    
    def dense(self):
        """
        Finds a standard GaussianDistribution object representing the same
        random variate as this distribution does. This involves creating a
        dense version of the covariance matrix, which may be prohibitive if the
        dimension is large enough compared to system memory.
        
        returns: a GaussianDistribution object
        """
        return GaussianDistribution(self.mean, self.covariance.toarray(),\
            metadata=self.metadata)
    
    def __add__(self, other):
        """
        Adds the Gaussian random variate described by this distribution
        to the given object.
        
        other: if other is a constant, the returned Gaussian is the same as
                                       this one with other added to the mean
                                       and the same covariance
               if other is a 1D numpy.ndarray, it must be of the same length
                                               as the dimension of this
                                               GaussianDistribution. In this
                                               case, the returned
                                               GaussianDistribution is the
                                               distribution of the sum of this
                                               Gaussian variate with other
               if other is a SparseGaussianDistribution, it must have the same
                                                         number of parameters
                                                         as this one
        
        returns: SparseGaussianDistribution representing the addition of this
                 Gaussian variate with other
        """
        if isinstance(other, SparseGaussianDistribution):
            if self.numparams == other.numparams:
                new_mean = self.mean + other.mean
                new_covariance_list =\
                    SparseGaussianDistribution.add_covariance_lists(\
                    self.covariance_list, other.covariance_list)
            else:
                raise ValueError("Cannot add together two " +\
                    "SparseGaussianDistribution objects with different " +\
                    "numbers of parameters.")
        elif type(other) in [list, tuple, np.ndarray]:
            other = np.array(other)
            if other.ndim == 1:
                if len(other) == self.numparams:
                    new_mean = self.mean + other
                    new_covariance_list = self.covariance_list
                else:
                    raise ValueError("Cannot multiply Gaussian distributed " +\
                        "random vector by a vector of different size.")
            else:
                raise ValueError("Cannot multiply Gaussian distributed " +\
                    "random vector by a tensor with more than 1 index.")
        else:
            # assume other is a constant
            new_mean = self.internal_mean.A[0] + other
            new_covariance_list = self.covariance_list
        return SparseGaussianDistribution(new_mean, new_covariance_list)
    
    @staticmethod
    def add_covariance_lists(list1, list2):
        """
        Adds the two covariance_lists to create a semi-efficient (efficient on
        average, but not guaranteed to be most efficient, especially if some
        covariances are exactly cancelled between list1 and list2)
        covariance_list that represents the sum of the covariance matrices
        represented by the given covariance_lists.
        
        list1, list2: covariance_list properties of SparseGaussianDistribution
                      objects to combine into one
        
        returns: a covariance_list processable in the __init__ method of the
                 SparseGaussianDistribution class representing the sum of the
                 covariances matrices represented by list1 and list2
        """
        if sum([len(element) for element in list1]) !=\
            sum([len(element) for element in list2]):
            raise ValueError("The two covariance_lists cannot be added " +\
                "because they correspond to different size covariance " +\
                "matrices.")
        (index1, index2) = (0, 0)
        (block1, block2) = (np.array([[]]), np.array([[]]))
        current = []
        while True:
            if block1.size < max(1, block2.size):
                if block1.size == 0:
                    block1 = list1[index1]
                else:
                    block1 = scila.block_diag(block1, list1[index1])
                index1 += 1
            elif block2.size < block1.size:
                if block2.size == 0:
                    block2 = list2[index2]
                else:
                    block2 = scila.block_diag(block2, list2[index2])
                index2 += 1
            if block1.size == block2.size:
                current.append(block1 + block2)
                (block1, block2) = (np.array([[]]), np.array([[]]))
            if (block1.size + block2.size) == 0:
                if (index1 == len(list1)) or (index2 == len(list2)):
                    break
        return current
    
    def __radd__(self, other):
        """
        Returns the same thing as __add__ (this makes addition commutative).
        """
        return self.__add__(other)
    
    def __sub__(self, other):
        """
        Returns the same thing as __add__ with an argument of (-other)
        """
        return self.__add__(-other)
    
    def __rsub__(self, other):
        """
        Returns the negative of the returned value of __sub__ (this makes
        subtraction doable from either side).
        """
        return self.__sub__(other).__neg__()
    
    def __neg__(self):
        """
        Returns a new GaussianDistribution with a mean multiplied by -1.
        """
        return SparseGaussianDistribution(-self.mean, self.covariance_list)
    
    def __mul__(self, other):
        """
        Multiplies the Gaussian random variate described by this distribution
        by the given constant.
        
        other: a constant number
        
        returns: GaussianDistribution representing the product of the random
                 variable with this distribution and the constant other
        """
        new_mean = self.mean * other
        new_covariance_list =\
            [(block * (other ** 2)) for block in self.covariance_list]
        return SparseGaussianDistribution(new_mean, new_covariance_list)
    
    def __rmul__(self, other):
        """
        Returns the same thing as __mul__ (this makes multiplication
        commutative).
        """
        return self.__mul__(other)
    
    def __div__(self, other):
        """
        Returns the same thing as __mul__ with an argument of (1/other).
        """
        return self.__mul__(1 / other)
    
    def __matmul__(self, other):
        """
        Finds and returns the distribution of the matrix product of other with
        the random variable this distribution describes.
        
        other: if other is a 1D numpy.ndarray, it must be of the same length
                                               as the dimension of this
                                               GaussianDistribution. In this
                                               case, the returned
                                               GaussianDistribution is the
                                               distribution of the dot product
                                               of this Gaussian variate with
                                               other
               if other is a 2D numpy.ndarray, it must have shape
                                               (newparams, self.numparams)
                                               where newparams<=self.numparams
                                               The returned
                                               GaussianDistribution is the
                                               distribution of other (matrix)
                                               multiplied with this Gaussian
                                               variate
        
        returns: GaussianDistribution object (note that this doesn't return a
                 SparseGaussianDistribution object; restrictions would have to
                 be placed on the matrix forms of other when it is 2D that are
                 too complicated)
        """
        if type(other) in [list, tuple, np.ndarray]:
            other = np.array(other)
            if other.ndim == 1:
                if len(other) == self.numparams:
                    new_mean = np.dot(self.mean, other)
                    new_covariance =\
                        np.dot(self.covariance.dot(other), other)
                else:
                    raise ValueError("Cannot multiply Gaussian distributed " +\
                        "random vector by a vector of different size.")
            elif other.ndim == 2:
                if other.shape[1] == self.numparams:
                    if other.shape[0] <= self.numparams:
                        # other is a matrix with self.numparams columns
                        new_mean = np.dot(other, self.mean)
                        new_covariance =\
                            np.dot(other, self.covariance.dot(other.T))
                    else:
                        raise ValueError("Cannot multiply Gaussian " +\
                            "distributed random vector by matrix which " +\
                            "will expand the number of parameters because " +\
                            "the covariance matrix of the result would be " +\
                            "singular.")
                else:
                    raise ValueError("Cannot multiply given matrix with " +\
                        "Gaussian distributed random vector because the " +\
                        "axis of its second dimension is not the same " +\
                        "length as the random vector.")
            else:
                raise ValueError("Cannot multiply Gaussian distributed " +\
                    "random vector by a tensor with more than 2 indices.")
        else:
            raise TypeError("Matrix multiplication can only be done with " +\
                "sequence types.")
        return GaussianDistribution(new_mean, new_covariance)
    
    def draw(self, shape=None, random=rand):
        """
        Draws a point from this distribution using numpy.random.
        
        shape: if None, returns single random variate
                        (scalar for univariate ; 1D array for multivariate)
               if int, n, returns n random variates
                          (1D array for univariate ; 2D array for multivariate)
               if tuple of n ints, returns that many random variates
                                   n-D array for univariate ;
                                   (n+1)-D array for multivariate
        random: the random number generator to use (default: numpy.random)
        
        returns a numpy.ndarray containing the values from this draw
        """
        if type(shape) is type(None):
            return self.mean + self.square_root_covariance.dot(\
                random.normal(0, 1, size=self.numparams))
        else:
            if type(shape) in int_types:
                shape = (shape,)
            flattened_shape = (self.numparams, np.prod(shape))
            random_vector = random.normal(0, 1, size=flattened_shape)
            values = self.mean[:,np.newaxis] +\
                self.square_root_covariance.dot(random_vector)
            return np.reshape(values.T, shape + (self.numparams,))
    
    @property
    def log_value_constant_part(self):
        """
        Property storing the constant part of the log value, i.e. the part of
        the sum that has no dependence on the point at which the distribution
        is being evaluated.
        """
        if not hasattr(self, '_log_value_constant_part'):
            self._log_value_constant_part = (self.log_determinant_covariance +\
            (self.numparams * natural_log_two_pi)) / (-2.)
        return self._log_value_constant_part
    
    def log_value(self, point):
        """
        Evaluates the log of the value of this distribution at the given point.
        
        point: numpy.ndarray of shape (self.numparams,)
        
        returns: the log of the value of this distribution at the given point
        """
        minus_mean = point - self.mean
        exponent =\
            np.dot(minus_mean, self.inverse_covariance.dot(minus_mean)) / (-2.)
        return self.log_value_constant_part + exponent
    
    def to_string(self):
        """
        Finds and returns the string representation of this
        SparseGaussianDistribution.
        """
        return "{}-dim Sparse Normal".format(self.numparams)
    
    def __eq__(self, other):
        """
        Checks for equality of this distribution with other. Returns True if
        other is a GaussianDistribution with the same mean (down to 10^-9
        level) and variance (down to 10^-12 dynamic range) and False otherwise.
        Note that this will return false if other is a GaussianDistribution
        even if it represents the same distribution as this
        SparseGaussianDistribution.
        """
        if isinstance(other, SparseGaussianDistribution):
            if self.numparams == other.numparams:
                mean_close =\
                    np.allclose(self.mean, other.mean, rtol=0, atol=1e-9)
                covariance_close =\
                    SparseGaussianDistribution.covariance_lists_close(\
                    self.covariance_list, other.covariance_list, rtol=1e-12,\
                    atol=0)
                metadata_equal = self.metadata_equal(other)
                return all([mean_close, covariance_close, metadata_equal])
            else:
                return False
        else:
            return False
    
    @staticmethod
    def covariance_lists_close(list1, list2, **allclose_kwargs):
        """
        Checks if the given covariance_lists (which have been processed by the
        __init__ method of the SparseGaussianDistribution class) are
        equivalent up to the given tolerance.
        
        list1, list2: covariance_list properties of SparseGaussianDistribution
                      objects
        **allclose_kwargs: keyword arguments to pass to numpy.allclose function
                           to determine tolerance. Generally, this should
                           include rtol and/or atol to represent relative and
                           absolute tolerances.
        
        returns: True if list1 and list2 represent the same covariance matrices
                 (even if represented differently), False otherwise
        """
        if sum([len(element) for element in list1]) !=\
            sum([len(element) for element in list2]):
            return False
        (index1, index2) = (0, 0)
        (block1, block2) = (np.array([[]]), np.array([[]]))
        while True:
            if block1.size < max(1, block2.size):
                if block1.size == 0:
                    block1 = list1[index1]
                else:
                    block1 = scila.block_diag(block1, list1[index1])
                index1 += 1
            elif block2.size < block1.size:
                if block2.size == 0:
                    block2 = list2[index2]
                else:
                    block2 = scila.block_diag(block2, list2[index2])
                index2 += 1
            if block1.size == block2.size:
                if not np.allclose(block1, block2, **allclose_kwargs):
                    return False
                (block1, block2) = (np.array([[]]), np.array([[]]))
            if (block1.size + block2.size) == 0:
                if (index1 == len(list1)) or (index2 == len(list2)):
                    return True
    
    @property
    def minimum(self):
        """
        Property storing the minimum allowable values in this distribution.
        """
        return ([None] * self.numparams)
    
    @property
    def maximum(self):
        """
        Property storing the maximum allowable values in this distribution.
        """
        return ([None] * self.numparams)
    
    @property
    def is_discrete(self):
        """
        Property storing a boolean describing whether this distribution is
        discrete (True) or continuous (False).
        """
        return False
    
    def fill_hdf5_group(self, group, mean_link=None, covariance_links=None,\
        save_metadata=True):
        """
        Fills the given hdf5 file group with data from this distribution. The
        fact that this is a GaussianDistribution is saved along with the mean
        and covariance.
        
        group: hdf5 file group to fill
        mean_link: link to mean already existing in file (if it exists)
        covariance_links: links to covariance blocks already existing in file
                          (if they exists)
        save_metadata: if True, attempts to save metadata alongside
                                distribution and throws error if it fails
                       if False, metadata is ignored in saving process
        """
        group.attrs['class'] = 'SparseGaussianDistribution'
        create_hdf5_dataset(group, 'mean', data=self.mean, link=mean_link)
        this_link = None
        subgroup = group.create_group('covariance')
        for (index, block) in enumerate(self.covariance_list):
            if type(covariance_links) is not type(None):
                this_link = covariance_links[index]
            create_hdf5_dataset(subgroup, 'block_{:d}'.format(index),\
                data=block, link=this_link)
        if save_metadata:
            self.save_metadata(group)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a GaussianDistribution from the given hdf5 file group.
        
        group: the same hdf5 file group which fill_hdf5_group was called on
               when this Distribution was saved
        
        returns: a SparseGaussianDistribution object created from the
                 information in the given group
        """
        try:
            assert group.attrs['class'] == 'SparseGaussianDistribution'
        except:
            raise TypeError("The given hdf5 file doesn't seem to contain a " +\
                "SparseGaussianDistribution.")
        metadata = Distribution.load_metadata(group)
        mean = get_hdf5_value(group['mean'])
        (covariance_list, current_index) = ([], 0)
        subgroup = group['covariance']
        while 'block_{:d}'.format(current_index) in subgroup:
            covariance_list.append(\
                get_hdf5_value(subgroup['block_{:d}'.format(current_index)]))
            current_index += 1
        return SparseGaussianDistribution(mean, covariance_list,\
            metadata=metadata)
    
    @property
    def gradient_computable(self):
        """
        Property which stores whether the gradient of the given distribution
        has been implemented. Since it has been implemented, it returns True.
        """
        return True
    
    def gradient_of_log_value(self, point):
        """
        Computes the derivatives of log_value(point) with respect to the
        parameters.
        
        point: 1D vector at which to evaluate the derivatives
        
        returns: if this Gaussian is 1D, returns single value of derivative
                 else, returns 1D vector of values of derivatives
        """
        return self.inverse_covariance.dot(self.mean - point)
    
    @property
    def hessian_computable(self):
        """
        Property which stores whether the hessian of the given distribution
        has been implemented. Since it has been implemented, it returns True.
        """
        return True
    
    def hessian_of_log_value(self, point):
        """
        Computes the second derivatives of log_value(point) with respect to the
        parameters.
        
        point: vector at which to evaluate the derivatives
        
        returns: 2D square matrix of second derivatives of log value
        """
        return -self.inverse_covariance.toarray()
    
    def copy(self):
        """
        Returns a deep copy of this Distribution. This function ignores
        metadata.
        """
        return SparseGaussianDistribution(self.mean.copy(),\
            [block.copy() for block in self.covariance_list])

