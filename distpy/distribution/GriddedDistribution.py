"""
File: distpy/GriddedDistribution.py
Author: Keith Tauscher
Date: 6 Aug 2017

Description: File containing class representing an arbitrary dimensional
             distribution given by a rectangular array-defined pdf.
"""
import numpy as np
import numpy.random as rand
from ..util import int_types, sequence_types, numerical_types,\
    create_hdf5_dataset
from .Distribution import Distribution


def search_sorted(array, value):
    """
    Searches the given sorted array for the given value using a
    BinarySearch which should execute in O(log N).
    
    array a 1D sorted numerical array
    value the numerical value to search for

    returns index of array closest to value
            returns None if value is outside variable bounds
    """
    def index_to_check(rmin, rmax):
        return (rmin + rmax) // 2
    range_min = 0
    range_max_0 = len(array)
    range_max = range_max_0
    numloops = 0
    while numloops < 100:
        numloops += 1
        if (range_max - range_min) == 1:
            if (range_max == range_max_0) or (range_min == 0):
                raise LookupError(("For some reason, range_max-" +\
                    "range_min reached 1 before the element was found. The " +\
                    "element being searched for was {0!s}. (min,max)=" +\
                    "({1!s},{2!s})").format(value, range_min, range_max))
            else:
                high_index = range_max
        else:
            high_index = index_to_check(range_min, range_max)
        high_val = array[high_index]
        low_val = array[high_index - 1]
        if value < low_val:
            range_max = high_index
        elif value > high_val:
            range_min = high_index
        else: # low_val <= value <= high_val
            if (2 * (high_val - value)) < (high_val - low_val):
                return high_index
            else:
                return high_index - 1
    raise NotImplementedError("Something went wrong! I got " +\
        "caught a pseudo-infinite loop!")
        

class GriddedDistribution(Distribution):
    """
    A class representing an arbitrary dimensional (well, up to 32 dimensions)
    probability distribution (of finite support!).
    """
    def __init__(self, variables, pdf=None, metadata=None):
        """
        Initializes a new GriddedDistribution using the given variables.
        
        variables list of variable ranges (i.e. len(variables) == ndim
                  and variables[i] is the set of the ith variables)
        pdf numpy.ndarray with same ndim as number of parameters and with
              the ith axis having the same length as the ith variables range
        """
        if type(variables) in sequence_types:
            self._N = len(variables)
            self.vars = [variables[i] for i in range(len(variables))]
            self.shape =\
                tuple([len(variables[i]) for i in range(self.numparams)])
            self.size = np.prod(self.shape)
            if pdf is None:
                self.pdf = np.ones(self.shape)
            elif type(pdf) in sequence_types:
                arrpdf = np.array(pdf)
                if arrpdf.shape == self.shape:
                    self.pdf = arrpdf
                else:
                    raise ValueError(("The pdf given to a " +\
                        "GriddedDistribution were not of the expected " +\
                        "shape. It should be an N-dimensional array with " +\
                        "each dimension given by the length of the " +\
                        "corresponding variable's range. Its values should " +\
                        "be proportional to the pdf. The shape was {0!s} " +\
                        "when it should have been {1!s}.").format(\
                        arrpdf.shape, self.shape))
            else:
                raise ValueError("The pdf given to a GriddedDistribution " +\
                    "were not of a sequence type. It should be an " +\
                    "N-dimensional array with each dimension given by the " +\
                    "length of the corresponding variable's range. Its " +\
                    "values should be proportional to the pdf.")
        else:
            raise ValueError("The variables given to a GriddedDistribution " +\
                "were not of a list type. It should be a sequence of " +\
                "variable ranges.")
        self.pdf = self.pdf.flatten()
        self._make_cdf()
        self.metadata = metadata

    @property
    def numparams(self):
        """
        Finds and returns the number of parameters which this distribution
        describes.
        """
        return self._N

    def draw(self, shape=None):
        """
        Draws and returns a point from this distribution.
        
        shape: if None, returns single random variate
                        (scalar for univariate ; 1D array for multivariate)
               if int, n, returns n random variates
                          (1D array for univariate ; 2D array for multivariate)
               if tuple of n ints, returns that many random variates
                                   n-D array for univariate ;
                                   (n+1)-D array for multivariate
        """
        if shape is None:
            shape = ()
        if type(shape) in int_types:
            shape = (shape,)
        return self.inverse_cdf(rand.rand(*shape))
    
    def inverse_cdf(self, cdf):
        """
        Inverse of the cumulative distribution function (only expected to work
        for if self.numparams == 1 but works nonetheless in higher dimensions).
        
        cdf: value between 0 and 1
        """
        if type(cdf) in numerical_types:
            inv_cdf_index = self._inverse_cdf_by_packed_index(cdf)
            return self._point_from_packed_index(inv_cdf_index)
        else:
            cdf = np.array(cdf)
            if self.numparams == 1:
                return_val = np.ndarray(cdf.shape)
            else:
                return_val = np.ndarray(cdf.shape + (self.numparams,))
            for multi_index in np.ndindex(*cdf.shape):
                inv_cdf_index =\
                    self._inverse_cdf_by_packed_index(cdf[multi_index])
                return_val[multi_index] =\
                    self._point_from_packed_index(inv_cdf_index)
            return return_val

    def log_value(self, point):
        """
        Evaluates and returns the log of the pdf of this distribution at the
        given point.
        
        point: numpy.ndarray of variable values describing the point
        
        returns the log of the pdf associated with the pixel containing point
        """
        index = self._packed_index_from_point(point)
        if index is None:
            return -np.inf
        return np.log(self.pdf[index])


    def to_string(self):
        """
        Finds and returns a string representation of this GriddedDistribution.
        """
        return "Gridded(user defined)"
    
    def __eq__(self, other):
        """
        Checks for equality of this distribution with other. Returns True if
        other is a GriddedDistribution with the same variable ranges and pdf
        and False otherwise.
        """
        if isinstance(other, GriddedDistribution):
            if self.numparams == other.numparams:
                if self.shape == other.shape:
                    vars_close =\
                        np.allclose(self.vars, other.vars, rtol=0, atol=1e-9)
                    pdf_close =\
                        np.allclose(self.pdf, other.pdf, rtol=0, atol=1e-12)
                    metadata_equal = self.metadata_equal(other)
                    return all([vars_close, pdf_close, metadata_equal])
                else:
                    return False
            else:
                return False
        else:
            return False

    def _make_cdf(self):
        #
        # Constructs the cdf array.
        #
        running_sum = 0.
        print('initializing cdf')
        self.cdf = np.ndarray(len(self.pdf))
        print('filling cdf')
        for i in range(len(self.pdf)):
            self.cdf[i] = running_sum
            running_sum += (self.pdf[i] * self._pixel_area(i))
        print('renormalizing pdf and cdf')
        self.cdf = self.cdf / self.cdf[-1]
        self.pdf = self.pdf / self.cdf[-1]

    def _unpack_index(self, index):
        #
        # Finds N-dimensional index corresponding to given index.
        #
        if index is None:
            return None
        inds_in_reverse = []
        running_product = self.shape[self._N - 1]
        inds_in_reverse.append(index % running_product)
        for k in range(1, self._N):
            rel_dim = self.shape[self._N - k - 1]
            inds_in_reverse.append((index // running_product) % rel_dim)
            running_product *= rel_dim
        return inds_in_reverse[-1::-1]

    def _pack_index(self, unpacked_indices):
        #
        # Finds single index which is the packed version
        # of unpacked_indices (which should be a list)
        #
        if unpacked_indices is None:
            return None
        cumulative_index = 0
        running_product = 1
        for i in range(self._N - 1, - 1, - 1):
            cumulative_index += (running_product * unpacked_indices[i])
            running_product *= self.shape[i]
        return cumulative_index

    def _unpacked_indices_from_point(self, point):
        #
        # Gets the unpacked indices which is associated with this point.
        #
        unpacked_indices = []
        for ivar in range(self._N):
            try:
                index = search_sorted(self.vars[ivar], point[ivar])
            except LookupError:
                return None
            unpacked_indices.append(index)
        return unpacked_indices

    def _packed_index_from_point(self, point):
        #
        # Finds the packed index associated with the given point.
        #
        return self._pack_index(self._unpacked_indices_from_point(point))
        
    def _point_from_packed_index(self, index):
        #
        # Finds the point associated with the given packed index
        #
        int_part = int(index + 0.5)
        unpacked_indices = self._unpack_index(int_part)
        point = [self.vars[i][unpacked_indices[i]] for i in range(self._N)]
        return  np.array(point) + self._continuous_offset(unpacked_indices)

    def _continuous_offset(self, unpacked_indices):
        #
        # Finds a vector offset to simulate a continuous distribution (even
        # though, internally pixels are being used
        #
        return np.array(\
            [self._single_var_offset(i, unpacked_indices[i], rand.rand())\
             for i in range(self._N)])

    def _single_var_offset(self, ivar, index, rval):
        #
        # Finds the offset for a single variable. rval should be Unif(0,1)
        #
        this_var_length = self.shape[ivar]
        if index == 0:
            return (0.5 *\
                    rval * (self.vars[ivar][1] - self.vars[ivar][0]))
        elif index == (this_var_length - 1):
            return ((-0.5) *\
                    rval * (self.vars[ivar][-1] - self.vars[ivar][-2]))
        else:
            return 0.5 * (self.vars[ivar][index]\
                          - (rval * self.vars[ivar][index - 1])\
                          - ((1 - rval) * self.vars[ivar][index + 1]))

    def _pixel_area(self, packed_index):
        #
        # Finds the area of the pixel described by the given index.
        #
        pixel_area = 1.
        unpacked_indices = self._unpack_index(packed_index)
        for ivar in range(self._N):
            this_index = unpacked_indices[ivar]
            if this_index == 0:
                pixel_area *= (0.5 * (self.vars[ivar][1] - self.vars[ivar][0]))
            elif this_index == len(self.vars[ivar]) - 1:
                pixel_area *= (0.5 *\
                               (self.vars[ivar][-1] - self.vars[ivar][-2]))
            else:
                pixel_area *= (0.5 * (self.vars[ivar][this_index + 1] -\
                                      self.vars[ivar][this_index - 1]))
        return pixel_area

    def _inverse_cdf_by_packed_index(self, value):
        #
        # Finds the index where the cdf has the given value.
        #
        return search_sorted(self.cdf, value)
    
    def fill_hdf5_group(self, group, pdf_link=None):
        """
        Fills the given hdf5 file group with data from this distribution. The
        class name, variables list, and pdf values.
        
        group: hdf5 file group to fill
        """
        group.attrs['class'] = 'GriddedDistribution'
        group.attrs['numparams'] = self.numparams
        for ivar in range(len(self.vars)):
            group.attrs['variable_{}'.format(ivar)] = self.vars[ivar]
        create_hdf5_dataset(group, 'pdf', data=self.pdf, link=pdf_link)
        self.save_metadata(group)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a GriddedDistribution from the given hdf5 file group.
        
        group: the same hdf5 file group which fill_hdf5_group was called on
               when this Distribution was saved
        
        returns: a GriddedDistribution object created from the information in
                 the given group
        """
        try:
            assert group.attrs['class'] == 'GriddedDistribution'
        except:
            raise TypeError("The given hdf5 file doesn't seem to contain a " +\
                "GriddedDistribution.")
        metadata = Distribution.load_metadata(group)
        variables = []
        ivar = 0
        while ('variable_{}'.format(ivar)) in group.attrs:
            variables.append(group.attrs['variable_{}'.format(ivar)])
            ivar += 1
        pdf = get_hdf5_value(group['pdf'])
        return GriddedDistribution(variables=variables, pdf=pdf,\
            metadata=metadata)
    
    @property
    def gradient_computable(self):
        """
        Property which stores whether the gradient of the given distribution
        has been implemented. It has not been implemented, so it returns False.
        """
        return False 
    
    @property
    def hessian_computable(self):
        """
        Property which stores whether the hessian of the given distribution
        has been implemented. It has not been implemented, so it returns False.
        """
        return False

