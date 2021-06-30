"""
Module containing class representing a distribution defined on a user-defined
grid. Within grid squares, the pdf is uniform.

**File**: $DISTPY/distpy/distribution/GriddedDistribution.py  
**Author**: Keith Tauscher  
**Date**: 31 May 2021
"""
import numpy as np
import numpy.random as rand
from ..util import int_types, sequence_types, numerical_types,\
    create_hdf5_dataset, get_hdf5_value
from .Distribution import Distribution


def search_sorted(array, value):
    """
    Searches the given sorted array for the given value using a binary search
    which should execute in O(log N).
    
    Parameters
    ----------
    array : `numpy.ndarray`
        a 1D sorted numerical array
    value : float
        the numerical value to search for
    
    Returns
    -------
    index : int
        - if `value` is between `array[0]` and `array[-1]`, then `index` is the
        integer index of `array` closest to `value`
        - if `value<array[0]` or `value > array[-1]`, then `index` is None
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
    Class representing a distribution defined on a user-defined grid. Within
    grid squares, the pdf is uniform.
    """
    def __init__(self, variables, pdf=None, metadata=None):
        """
        Initializes a new `GriddedDistribution` with the given parameter
        values.
        
        Parameters
        ----------
        variables : sequence
            sequence of 1D arrays of lengths \\(N_1,N_2,\\ldots,N_n\\) defining
            grid square edges
        pdf : None or `numpy.ndarray`
            - if `pdf` is None, each grid square is equally likely
            - otherwise, `pdf` should be an \\(n\\)-dimensional array giving
            the (possibly unnormalized) probability of landing in each grid
            square
        metadata : number or str or dict or `distpy.util.Savable.Savable`
            data to store alongside this distribution.
        """
        if type(variables) in sequence_types:
            self._N = len(variables)
            self.vars = [variables[i] for i in range(len(variables))]
            self.shape =\
                tuple([len(variables[i]) for i in range(self.numparams)])
            self.size = np.prod(self.shape)
            if type(pdf) is type(None):
                self.pdf = np.ones(self.shape)
            elif type(pdf) in sequence_types:
                arrpdf = np.array(pdf)
                if arrpdf.size == np.prod(self.shape):
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
        The number of parameters of this `GriddedDistribution`.
        """
        return self._N
    
    @property
    def mean(self):
        """
        The mean of the `GriddedDistribution` class is not implemented.
        """
        if not hasattr(self, '_mean'):
            raise NotImplementedError("mean is not implemented for the " +\
                "GriddedDistribution.")
        return self._mean
    
    @property
    def variance(self):
        """
        The variance of the `GriddedDistribution` class is not implemented.
        """
        if not hasattr(self, '_variance'):
            raise NotImplementedError("variance is not implemented for the " +\
                "GriddedDistribution.")
        return self._variance

    def draw(self, shape=None, random=rand):
        """
        Draws point(s) from this `GriddedDistribution`.
        
        Parameters
        ----------
        shape : int or tuple or None
            - if None, returns single random variate:
                - if this distribution is univariate, a scalar is returned
                - if this distribution describes \\(p\\) parameters, then a 1D
                array of length \\(p\\) is returned
            - if int, \\(n\\), returns \\(n\\) random variates:
                - if this distribution is univariate, a 1D array of length
                \\(n\\) is returned
                - if this distribution describes \\(p\\) parameters, then a 2D
                array of shape `(n,p)` is returned
            - if tuple of \\(n\\) ints, returns `numpy.prod(shape)` random
            variates:
                - if this distribution is univariate, an \\(n\\)-D array of
                shape `shape` is returned
                - if this distribution describes \\(p\\) parameters, then an
                \\((n+1)\\)-D array of shape `shape+(p,)` is returned
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
            shape = ()
        if type(shape) in int_types:
            shape = (shape,)
        return self.inverse_cdf(random.rand(*shape))
    
    def inverse_cdf(self, cdf):
        """
        Computes the inverse of the cumulative distribution function (cdf) of
        this `GriddedDistribution`. Only works when
        `GriddedDistribution.numparams` is 1.
        
        Parameters
        ----------
        cdf : float
            probability value between 0 and 1
        
        Returns
        -------
        point : float
            value which yields `cdf` when it the CDF is evaluated at it
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
        Computes the logarithm of the value of this `GriddedDistribution` at
        the given point.
        
        Parameters
        ----------
        point : float or `numpy.ndarray`
            - if this distribution is univariate, `point` should be a scalar
            - if this distribution describes \\(p\\) parameters, `point` should
            be a length-\\(p\\) `numpy.ndarray`
        
        Returns
        -------
        value : float
            natural logarithm of the value of this distribution at `point`. If
            \\(f\\) is this distribution's PDF and \\(x\\) is `point`, then
            `value` is \\(\\ln{\\big(f(x)\\big)}\\)
        """
        index = self._packed_index_from_point(point)
        if (type(index) is type(None)) or (self.pdf[index] == 0):
            return -np.inf
        return np.log(self.pdf[index])

    def to_string(self):
        """
        Finds and returns a string version of this `GriddedDistribution` of the
        form `"Gridded(user defined)"`.
        """
        return "Gridded(user defined)"
    
    def __eq__(self, other):
        """
        Checks for equality of this `GriddedDistribution` with `other`.
        
        Parameters
        ----------
        other : object
            object to check for equality
        
        Returns
        -------
        result : bool
            True if and only if `other` is a `GriddedDistribution` with the
            same custom grid
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
        self.cdf = np.ndarray(len(self.pdf))
        for i in range(len(self.pdf)):
            self.cdf[i] = running_sum
            running_sum += (self.pdf[i] * self._pixel_area(i))
        self.cdf = self.cdf / self.cdf[-1]
        self.pdf = self.pdf / self.cdf[-1]

    def _unpack_index(self, index):
        #
        # Finds N-dimensional index corresponding to given index.
        #
        if type(index) is type(None):
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
        if type(unpacked_indices) is type(None):
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
    
    @property
    def minimum(self):
        """
        The minimum allowable value(s) in this distribution.
        """
        return [np.min(var) for var in self.vars]
    
    @property
    def maximum(self):
        """
        The maximum allowable value(s) in this distribution.
        """
        return [np.max(var) for var in self.vars]
    
    @property
    def is_discrete(self):
        """
        Boolean describing whether this distribution is discrete (True) or
        continuous (False).
        """
        return False
    
    def fill_hdf5_group(self, group, pdf_link=None, save_metadata=True):
        """
        Fills the given hdf5 file group with data about this
        `GriddedDistribution` so that it can be loaded later.
        
        Parameters
        ----------
        group : h5py.Group
            hdf5 file group to fill
        pdf_link : str or h5py.Dataset or None
            link to pdf in hdf5 file, if it exists
        save_metadata : bool
            - if True, attempts to save metadata alongside distribution and
            throws error if it fails
            - if False, metadata is ignored in saving process
        """
        group.attrs['class'] = 'GriddedDistribution'
        group.attrs['numparams'] = self.numparams
        for ivar in range(len(self.vars)):
            group.attrs['variable_{}'.format(ivar)] = self.vars[ivar]
        create_hdf5_dataset(group, 'pdf', data=self.pdf, link=pdf_link)
        if save_metadata:
            self.save_metadata(group)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a `GriddedDistribution` from the given hdf5 file group.
        
        Parameters
        ----------
        group : h5py.Group
            the same hdf5 file group which fill_hdf5_group was called on when
            this Distribution was saved
        
        Returns
        -------
        distribution : `GriddedDistribution`
            distribution created from the information in the given group
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
        Boolean describing whether the gradient of the given distribution has
        been implemented. If True,
        `GriddedDistribution.gradient_of_log_value` method can be called
        safely.
        """
        return False 
    
    @property
    def hessian_computable(self):
        """
        Boolean describing whether the hessian of the given distribution has
        been implemented. If True,
        `GriddedDistribution.hessian_of_log_value` method can be called
        safely.
        """
        return False
    
    def copy(self):
        """
        Copies this distribution.
        
        Returns
        -------
        copied : `GriddedDistribution`
            a deep copy of this distribution, ignoring metadata.
        """
        return GriddedDistribution(self.vars, self.pdf)

