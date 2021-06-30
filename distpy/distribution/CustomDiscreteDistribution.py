"""
Module containing class representing a custom discrete distribution. Its PMF is
represented by: $$f(x)=\\begin{cases} p_1 & x=x_1 \\\\ p_2 & x=x_2 \\\\\
\\vdots & \\vdots \\\\ p_N & x=x_N\\end{cases},$$ where
\\(\\sum_{k=1}^Np_k=1\\).

**File**: $DISTPY/distpy/distribution/CustomDiscreteDistribution.py  
**Author**: Keith Tauscher  
**Date**: 31 May 2021
"""
import numpy as np
import numpy.random as rand
from ..util import int_types, sequence_types, create_hdf5_dataset,\
    get_hdf5_value
from .Distribution import Distribution

class CustomDiscreteDistribution(Distribution):
    """
    Class representing a custom discrete distribution. Its PMF is represented
    by: $$f(x)=\\begin{cases} p_1 & x=x_1 \\\\ p_2 & x=x_2 \\\\ \\vdots &\
    \\vdots \\\\ p_N & x=x_N\\end{cases},$$ where \\(\\sum_{k=1}^Np_k=1\\).
    """
    def __init__(self, variable_values, probability_mass_function,\
        metadata=None):
        """
        Initializes a new `CustomDiscreteDistribution` with the given parameter
        values.
        
        Parameters
        ----------
        variable_values : sequence
            sequence of 1D arrays of length \\(N_1,N_2,\\ldots,N_n\\) that
            define the grids in an arbitrary number of variables, \\(n\\)
        probability_mass_function : `numpy.ndarray`
            \\(n\\)-dimensional array of shape \\((N_1,N_2,\\ldots,N_n)\\)
            containing (possibly unnormalized) probability masses of each grid
            point
        metadata : number or str or dict or `distpy.util.Savable.Savable`
            data to store alongside this distribution.
        """
        self.variable_values = variable_values
        self.probability_mass_function = probability_mass_function
        self.metadata = metadata
    
    @property
    def variable_values(self):
        """
        A sequence of 1D `numpy.ndarray` objects each containing the possible
        values of each of this distribution's variables.
        """
        if not hasattr(self, '_variable_values'):
            raise AttributeError("variable_values referenced before it was " +\
                "set.")
        return self._variable_values
    
    @variable_values.setter
    def variable_values(self, value):
        """
        Setter for `CustomDiscreteDistribution.variable_values`.
        
        Parameters
        ----------
        value : numpy.ndarray
            - if this distribution is 1D, `value` can be either a single 1D
            numpy.ndarray or a list containing a single 1D numpy.ndarray.
            - if this distribution is \\(n\\) dimensional, then `value` must be
            a length-\\(n\\) sequence of 1D numpy.ndarrays giving the possible
            values of the variables described by this distribution
        """
        if isinstance(value, np.ndarray) and (value.ndim == 1):
            value = [value]
        if type(value) in sequence_types:
            if all([isinstance(element, np.ndarray) for element in value]):
                if all([(element.ndim == 1) for element in value]):
                    self._variable_values = [element for element in value]
                else:
                    raise TypeError("Not all numpy.ndarrays in " +\
                        "variable_values were 1 dimensional.")
            else:
                raise TypeError("Not all elements of variable_values were "+\
                    "numpy.ndarrays.")
        else:
            raise TypeError("variable_values was set to neither a 1D " +\
                "numpy.ndarray or a sequence of 1D numpy.ndarrays")
    
    @property
    def shape(self):
        """
        The shape of the grid used for this `CustomDiscreteDistribution` as a
        tuple. It should be the shape of the
        `CustomDiscreteDistribution.probability_mass_function` array.
        """
        if not hasattr(self, '_shape'):
            self._shape =\
                sum([(len(axis),) for axis in self.variable_values], ())
        return self._shape
    
    @property
    def probability_mass_function(self):
        """
        An \\(n\\)-dimensional numpy.ndarray of shape given by the sum of the
        shapes of the variable_values storing the probabilities of each
        possible combination of variables.
        """
        if not hasattr(self, '_probability_mass_function'):
            raise AttributeError("probability_mass_function was referenced " +\
                "before it was set.")
        return self._probability_mass_function
    
    @probability_mass_function.setter
    def probability_mass_function(self, value):
        """
        Setter for `CustomDiscreteDistribution.probability_mass_function`.
        
        Parameters
        ----------
        value : numpy.ndarray
            an array whose shape is given by `CustomDiscreteDistribution.shape`
            containing non-negative numbers representing unnormalized
            probabilities
        """
        if isinstance(value, np.ndarray):
            if value.shape == self.shape:
                unnormalized_cmf = np.cumsum(value.flatten())
                norm_factor = 1. / unnormalized_cmf[-1]
                self._probability_mass_function = value * norm_factor
                self._flattened_cumulative_mass_function =\
                    unnormalized_cmf * norm_factor
            else:
                raise ValueError(("The shape of the given " +\
                    "probability_mass_function ({0!s}) was not the " +\
                    "expected shape ({1!s}).").format(value.shape, self.shape))
        else:
            raise TypeError("probability_mass_function was not a " +\
                "numpy.ndarray.")
    
    @property
    def mean(self):
        """
        The mean of this `CustomDiscreteDistribution`.
        """
        if not hasattr(self, '_mean'):
            mean = np.ndarray((self.numparams,))
            for index in range(self.numparams):
                variable_slice = (index * (np.newaxis,)) + (slice(None),) +\
                    ((self.numparams - index - 1) * (np.newaxis,))
                mean[index] = np.sum(\
                    self.variable_values[index][variable_slice] *\
                    self.probability_mass_function)
            if self.numparams == 1:
                self._mean = mean[0]
            else:
                self._mean = mean
        return self._mean
    
    @property
    def variance(self):
        """
        The (co)variance of this `CustomDiscreteDistribution`.
        """
        if not hasattr(self, '_variance'):
            if self.numparams == 1:
                expected_square = np.sum((self.variable_values[0] ** 2) *\
                    self.probability_mass_function)
                self._variance = expected_square - (self.mean ** 2)
            else:
                expected_squares = np.ndarray((self.numparams,) * 2)
                for index1 in range(self.numparams):
                    variable_slice1 =\
                        (index1 * (np.newaxis,)) + (slice(None),) +\
                        ((self.numparams - index1 - 1) * (np.newaxis,))
                    values1 = self.variable_values[index1][variable_slice1]
                    for index2 in range(index1, self.numparams):
                        variable_slice2 =\
                            (index2 * (np.newaxis,)) + (slice(None),) +\
                            ((self.numparams - index2 - 1) * (np.newaxis,))
                        values2 = self.variable_values[index2][variable_slice2]
                        expected_square = np.sum(values1 * values2 *\
                            self.probability_mass_function)
                        expected_squares[index1,index2] = expected_square
                        expected_squares[index2,index1] = expected_square
                self._variance = expected_squares -\
                    (self.mean[:,np.newaxis] * self.mean[np.newaxis,:])
        return self._variance
    
    @property
    def flattened_cumulative_mass_function(self):
        """
        The flattened, normalized, cumulative mass function
        corresponding to
        `CustomDiscreteDistribution.probability_mass_function`. This array is
        importantly used in the process of drawing random values from this
        distribution.
        """
        if not hasattr(self, '_flattened_cumulative_mass_function'):
            raise AttributeError("flattened_cumulative_mass_function " +\
                "referenced before it was set.")
        return self._flattened_cumulative_mass_function
    
    def draw(self, shape=None, random=rand):
        """
        Draws point(s) from this `CustomDiscreteDistribution`.
        
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
        none_shape = (type(shape) is type(None))
        if none_shape:
            shape = 1
        if type(shape) in int_types:
            shape = (shape,)
        random_values = random.rand(*shape)
        flattened_locations = np.searchsorted(\
            self.flattened_cumulative_mass_function, random_values)
        unpacked_locations = np.unravel_index(flattened_locations, self.shape)
        draws = np.ndarray(shape + (self.numparams,))
        for (iparam, locations) in enumerate(unpacked_locations):
            draws[...,iparam] = self.variable_values[iparam][locations]
        if none_shape:
            draws = draws[0]
        if self.numparams == 1:
            return draws[...,0]
        else:
            return draws
    
    def log_value(self, point):
        """
        Computes the logarithm of the value of this
        `CustomDiscreteDistribution` at the given point.
        
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
        if self.numparams == 1:
            point = [point]
        multi_index = ()
        for (component, axis) in zip(point, self.variable_values):
            try:
                multi_index =\
                    multi_index + (np.where(component == axis)[0][0],)
            except:
                return -np.inf
        return np.log(self.probability_mass_function[multi_index])
    
    @property
    def gradient_computable(self):
        """
        Boolean describing whether the gradient of the given distribution has
        been implemented. If True,
        `CustomDiscreteDistribution.gradient_of_log_value` method can be called
        safely.
        """
        return False
    
    @property
    def hessian_computable(self):
        """
        Boolean describing whether the hessian of the given distribution has
        been implemented. If True,
        `CustomDiscreteDistribution.hessian_of_log_value` method can be called
        safely.
        """
        return False
    
    @property
    def numparams(self):
        """
        The number of parameters of this `CustomDiscreteDistribution`.
        """
        if not hasattr(self, '_numparams'):
            self._numparams = len(self.variable_values)
        return self._numparams
    
    def to_string(self):
        """
        Finds and returns a string version of this `CustomDiscreteDistribution`
        of the form `"d-dim custom discrete"`.
        """
        return "{}-dim custom discrete".format(self.numparams)
    
    def __eq__(self, other):
        """
        Checks for equality of this `CustomDiscreteDistribution` with `other`.
        
        Parameters
        ----------
        other : object
            object to check for equality
        
        Returns
        -------
        result : bool
            True if and only if `other` is a `CustomDiscreteDistribution` with
            the same custom probabilities
        """
        if isinstance(other, CustomDiscreteDistribution):
            if self.numparams == other.numparams:
                for iparam in range(self.numparams):
                    saxis = self.variable_values[iparam]
                    oaxis = other.variable_values[iparam]
                    if np.any(saxis != oaxis):
                        return False
                return np.allclose(self.probability_mass_function,\
                    other.probability_mass_function, rtol=0, atol=1e-10)
            else:
                return False
        else:
            return False
    
    @property
    def minimum(self):
        """
        The minimum allowable value(s) in this distribution.
        """
        if not hasattr(self, '_minimum'):
            self._minimum = np.array([np.min(varvals)\
                for varvals in self.variable_values])
        return self._minimum
    
    @property
    def maximum(self):
        """
        The maximum allowable value(s) in this distribution.
        """
        if not hasattr(self, '_maximum'):
            self._maximum = np.array([np.max(varvals)\
                for varvals in self.variable_values])
        return self._maximum
    
    @property
    def is_discrete(self):
        """
        Boolean describing whether this distribution is discrete (True) or
        continuous (False).
        """
        return True
    
    def fill_hdf5_group(self, group, save_metadata=True, pmf_link=None,\
        **axis_links):
        """
        Fills the given hdf5 file group with data about this
        `CustomDiscreteDistribution` so that it can be loaded later.
        
        Parameters
        ----------
        group : h5py.Group
            hdf5 file group to fill
        save_metadata : bool
            - if True, attempts to save metadata alongside distribution and
            throws error if it fails
            - if False, metadata is ignored in saving process
        pmf_link : str or None
            - if None, probability_mass_function is saved directly in the
            `"probability_mass_function"` group
            - otherwise, `pmf_link` should be a link to a extant location
            where the probability mass function is already stored
        axis_links : dict
            dictionary (default empty) whose keys are strings of the form
            `'variable{0:d}'.format(variable_number)` and whose values are
            either None or links to existing locations where the axis values
            are already saved
        """
        group.attrs['class'] = 'CustomDiscreteDistribution'
        create_hdf5_dataset(group, 'probability_mass_function',\
            data=self.probability_mass_function, link=pmf_link)
        subgroup = group.create_group('variable_values')
        for (iaxis, axis) in enumerate(self.variable_values):
            string_name = 'variable{0:d}'.format(iaxis)
            if string_name in axis_links:
                link = axis_links[string_name]
            else:
                link = None
            create_hdf5_dataset(subgroup, string_name, data=axis, link=link)
        if save_metadata:
            self.save_metadata(group)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a `CustomDiscreteDistribution` from the given hdf5 file group.
        
        Parameters
        ----------
        group : h5py.Group
            the same hdf5 file group which fill_hdf5_group was called on when
            this Distribution was saved
        
        Returns
        -------
        distribution : `CustomDiscreteDistribution`
            distribution created from the information in the given group
        """
        try:
            assert group.attrs['class'] == 'CustomDiscreteDistribution'
        except:
            raise TypeError("The given hdf5 file doesn't seem to contain a " +\
                "CustomDiscreteDistribution.")
        metadata = Distribution.load_metadata(group)
        variable_values = []
        ivar = 0
        subgroup = group['variable_values']
        while 'variable{0:d}'.format(ivar) in subgroup:
            axis = get_hdf5_value(subgroup['variable{0:d}'.format(ivar)])
            variable_values.append(axis)
            ivar += 1
        probability_mass_function =\
            get_hdf5_value(group['probability_mass_function'])
        return CustomDiscreteDistribution(variable_values,\
            probability_mass_function)
    
    @property
    def can_give_confidence_intervals(self):
        """
        Discrete distributions do not support confidence intervals.
        """
        return False
    
    def copy(self):
        """
        Copies this distribution.
        
        Returns
        -------
        copied : `CustomDiscreteDistribution`
            a deep copy of this distribution, ignoring metadata.
        """
        return CustomDiscreteDistribution(self.variable_values.copy(),\
            self.probability_mass_function.copy())

