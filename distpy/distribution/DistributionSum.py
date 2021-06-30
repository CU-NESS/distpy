"""
Module containing class representing a distribution that is a weighted sum of
other distributions. Its PDF is represented by: $$f(\\boldsymbol{x}) =\
\\frac{\\sum_{k=1}^Nw_kg_k(\\boldsymbol{x})}{\\sum_{k=1}^Nw_k},$$ where the
\\(w_k>0\\) and the \\(g_k\\) are PDFs (or PMFs) that define random variates of
the same dimension.

**File**: $DISTPY/distpy/distribution/DistributionSum.py  
**Author**: Keith Tauscher  
**Date**: 1 Jun 2021
"""
import numpy as np
import numpy.random as rand
from ..util import int_types, sequence_types, create_hdf5_dataset,\
    get_hdf5_value
from .Distribution import Distribution

class DistributionSum(Distribution):
    """
    Class representing a distribution that is a weighted sum of other
    distributions. Its PDF is represented by: $$f(\\boldsymbol{x}) =\
    \\frac{\\sum_{k=1}^Nw_kg_k(\\boldsymbol{x})}{\\sum_{k=1}^Nw_k},$$ where the
    \\(w_k>0\\) and the \\(g_k\\) are PDFs (or PMFs) that define random
    variates of the same dimension.
    """
    def __init__(self, distributions, weights, metadata=None):
        """
        Creates a new `DistributionSum` object out of the given distributions.
        
        Parameters
        ----------
        distributions : sequence
            sequence of `distpy.distribution.Distribution.Distribution` objects
            with the same numparams and PDFs given by \\(g_k\\)
        weights : sequence
            sequence of numbers, \\(w_k\\), with which to combine (need not be
            normalized but they must all be positive)
        """
        self.distributions = distributions
        self.weights = weights
        self.metadata = metadata
    
    @property
    def distributions(self):
        """
        A list of `distpy.distribution.Distribution.Distribution` objects which
        make up this `DistributionSum`. The PDFs are given by \\(g_k\\).
        """
        if not hasattr(self, '_distributions'):
            raise AttributeError("distributions was referenced before it " +\
                "was set.")
        return self._distributions
    
    @distributions.setter
    def distributions(self, value):
        """
        Setter for `DistributionSum.distributions`.
        
        Parameters
        ----------
        value : sequence
            sequence of `distpy.distribution.Distribution.Distribution` objects
        """
        if type(value) in sequence_types:
            if all([isinstance(element, Distribution) for element in value]):
                unique_nums_of_parameters =\
                    np.unique([element.numparams for element in value])
                unique_is_discretes =\
                    np.unique([element.is_discrete for element in value])
                if (len(unique_nums_of_parameters) == 1) and\
                    (len(unique_is_discretes) == 1):
                    self._distributions = [element for element in value]
                    self._numparams = unique_nums_of_parameters[0]
                    self._is_discrete = unique_is_discretes[0]
                else:
                    raise ValueError("Not all distributions added together " +\
                        "here had the same number of parameters.")
            else:
                raise TypeError("At least one element of the distributions " +\
                    "sequence was not a Distribution object.")
        else:
            raise TypeError("distributions was set to a non-sequence.")
    
    @property
    def num_distributions(self):
        """
        The number of `distpy.distribution.Distribution.Distribution` objects
        which make up this `DistributionSum` object.
        """
        if not hasattr(self, '_num_distributions'):
            self._num_distributions = len(self.distributions)
        return self._num_distributions
    
    @property
    def weights(self):
        """
        The weights, \\(w_k\\), with which to combine the underlying
        `distpy.distribution.Distribution.Distribution` objects.
        """
        if not hasattr(self, '_weights'):
            raise AttributeError("weights was referenced before it was set.")
        return self._weights
    
    @weights.setter
    def weights(self, value):
        """
        Setter for `DistributionSum.weights`.
        
        Parameters
        ----------
        value : sequence
            sequence of positive numbers
        """
        if type(value) in sequence_types:
            if len(value) == self.num_distributions:
                value = np.array(value)
                if np.all(np.zeros(self.num_distributions) < value):
                    self._weights = value
                else:
                    raise ValueError("At least one weight was non-positive.")
            else:
                raise ValueError("The length of the weights sequence was " +\
                    "not the same as the length of the distributions " +\
                    "sequence.")
        else:
            raise TypeError("weights was set to a non-sequence.")
    
    @property
    def total_weight(self):
        """
        The sum of all elements of `DistributionSum.weights`
        """
        if not hasattr(self, '_total_weight'):
            self._total_weight = np.sum(self.weights)
        return self._total_weight
    
    @property
    def minimum(self):
        """
        The minimum allowable value(s) in this distribution.
        """
        if not hasattr(self, '_minimum'):
            if self.numparams == 1:
                self._minimum = np.inf
                for distribution in self.distributions:
                    if type(distribution.minimum) is type(None):
                        self._minimum = None
                        break
                    else:
                        self._minimum =\
                            min(self._minimum, distribution.minimum)
            else:
                self._minimum = [np.inf] * self.numparams
                for distribution in self.distributions:
                    this_minimum = distribution.minimum
                    for iparam in range(self.numparams):
                        if type(self._minimum[iparam]) is type(None):
                            break
                        elif type(this_minimum[iparam]) is type(None):
                            self._minimum[iparam] = None
                            break
                        else:
                            self._minimum[iparam] = min(self._minimum[iparam],\
                                this_minimum[iparam])
        return self._minimum
    
    @property
    def maximum(self):
        """
        The maximum allowable value(s) in this distribution.
        """
        if not hasattr(self, '_maximum'):
            if self.numparams == 1:
                self._minimum = -np.inf
                for distribution in self.distributions:
                    if type(distribution.maximum) is type(None):
                        self._maximum = None
                        break
                    else:
                        self._maximum =\
                            max(self._maximum, distribution.maximum)
            else:
                self._maximum = [-np.inf] * self.numparams
                for distribution in self.distributions:
                    this_maximum = distribution.maximum
                    for iparam in range(self.numparams):
                        if type(self._maximum[iparam]) is type(None):
                            break
                        elif type(this_maximum[iparam]) is type(None):
                            self._maximum[iparam] = None
                            break
                        else:
                            self._maximum[iparam] = max(self._maximum[iparam],\
                                this_maximum[iparam])
        return self._maximum

    @property
    def mean(self):
        """
        The mean of the distribution,
        \\(\\frac{\\sum_{k=1}^Nw_k\\mu_k}{\\sum_{k=1}^Nw_k}\\), where
        \\(\\mu_k\\) is the mean of the \\(k^{\\text{th}}\\) element of
        `DistributionSum.distributions`.
        """
        if not hasattr(self, '_mean'):
            self._mean = np.sum([(weight * distribution.mean)\
                for (weight, distribution) in\
                zip(self.weights, self.distributions)]) / self.total_weight
        return self._mean
    
    @property
    def variance(self):
        """
        The (co)variance of the distribution,
        \\(\\frac{\\sum_{k=1}^Nw_k(\\mu_k\\mu_k^T+\\Sigma)}{\\sum_{k=1}^Nw_k}-\
        \\mu\\mu^T\\), where \\(\\mu_k\\) and \\(\\Sigma\\) are the mean and
        (co)variance of the \\(k^{\\text{th}}\\) element of
        `DistributionSum.distributions` and \\(\\mu\\) is
        `DistributionSum.mean`.
        """
        if not hasattr(self, '_variance'):
            if self.numparams == 1:
                expected_square = 0
                for (weight, distribution) in\
                    zip(self.weights, self.distributions):
                    this_term =\
                        distribution.variance + (distribution.mean ** 2)
                    expected_square = expected_square + (weight * this_term)
                expected_square = expected_square / self.total_weight
                self._variance = expected_square - (self.mean ** 2)
            else:
                expected_square = np.zeros(2 * (self.numparams,))
                for (weight, distribution) in\
                    zip(self.weights, self.distributions):
                    this_term = distribution.variance +\
                        (distribution.mean[:,np.newaxis] *\
                        distribution.mean[np.newaxis,:])
                    expected_square = expected_square + (weight * this_term)
                expected_square = expected_square / self.total_weight
                self._variance = expected_square -\
                    (self.mean[:,np.newaxis] * self.mean[np.newaxis,:])
        return self._variance
    
    @property
    def discrete_cdf_values(self):
        """
        The upper bound of CDF values allocated to each component of
        `DistributionSum.distributions`. The sequence is monotonically
        increasing and its last element is equal to 1.
        """
        if not hasattr(self, '_discrete_cdf_values'):
            self._discrete_cdf_values =\
                np.cumsum(self.weights) / self.total_weight
        return self._discrete_cdf_values
    
    def draw(self, shape=None, random=rand):
        """
        Draws point(s) from this `DistributionSum`.
        
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
            uniform = random.uniform()
            idistribution = np.argmax(self.discrete_cdf_values > uniform)
            distribution = self.distributions[idistribution]
            return distribution.draw(random=random)
        else:
            if type(shape) in int_types:
                shape = (shape,)
            size = np.prod(shape)
            uniforms = random.uniform(size=size)
            if self.numparams == 1:
                intermediate_shape = (size,)
                final_shape = shape
            else:
                intermediate_shape = (size, self.numparams)
                final_shape = shape + (self.numparams)
            draws = np.ndarray(intermediate_shape)
            cdf_upper_bound = 0
            for (idistribution, distribution) in enumerate(self.distributions):
                cdf_lower_bound = cdf_upper_bound
                cdf_upper_bound = self.discrete_cdf_values[idistribution]
                relevant_indices = np.nonzero((uniforms < cdf_upper_bound) &\
                    (uniforms >= cdf_lower_bound))[0]
                ndraw = len(relevant_indices)
                draws[relevant_indices,...] =\
                    distribution.draw(shape=ndraw, random=random)
            return np.reshape(draws, final_shape)
    
    def log_value(self, point):
        """
        Computes the logarithm of the value of this `DistributionSum` at the
        given point.
        
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
        values = np.exp([distribution.log_value(point)\
            for distribution in self.distributions])
        return np.log(np.sum(values * self.weights) / self.total_weight)
    
    @property
    def gradient_computable(self):
        """
        Boolean describing whether the gradient of the given distribution has
        been implemented. If True,
        `DistributionSum.gradient_of_log_value` method can be called safely.
        """
        return False
    
    @property
    def hessian_computable(self):
        """
        Boolean describing whether the hessian of the given distribution has
        been implemented. If True,
        `DistributionSum.hessian_of_log_value` method can be called safely.
        """
        return False
    
    @property
    def numparams(self):
        """
        The integer number of parameters described by this distribution.
        """
        if not hasattr(self, '_numparams'):
            raise AttributeError("numparams was referenced before " +\
                "distributions was set.")
        return self._numparams
    
    def to_string(self):
        """
        Finds a string representation of this distribution.

        Returns
        -------
        representation : str
            a string representation of this distribution of the form
            `"Sum of N dists"`
        """
        return "Sum of {:d} dists".format(self.num_distributions)
    
    def __eq__(self, other):
        """
        Checks for equality of this `DistributionSum` with `other`.
        
        Parameters
        ----------
        other : object
            object to check for equality
        
        Returns
        -------
        result : bool
            True if and only if `other` is a `DistributionSum` with the
            same `DistributionSum.distributions` and `DistributionSum.weights`
        """
        if isinstance(other, DistributionSum):
            if self.num_distributions == other.num_distributions:
                distributions_equal = all([(sdistribution == odistribution)\
                    for (sdistribution, odistribution) in\
                    zip(self.distributions, other.distributions)])
                weights_equal =\
                    np.allclose(self.weights, other.weights, rtol=1e-6, atol=0)
                return distributions_equal and weights_equal
            else:
                return False
        else:
            return False
    
    @property
    def is_discrete(self):
        """
        Boolean describing whether this distribution is discrete (True) or
        continuous (False).
        """
        if not hasattr(self, '_is_discrete'):
            raise AttributeError("is_discrete was referenced before " +\
                "distributions was set.")
        return self._is_discrete
    
    def fill_hdf5_group(self, group, save_metadata=True):
        """
        Fills the given hdf5 file group with data about this
        `DistributionSum` so that it can be loaded later.
        
        Parameters
        ----------
        group : h5py.Group
            hdf5 file group to fill
        save_metadata : bool
            - if True, attempts to save metadata alongside distribution and
            throws error if it fails
            - if False, metadata is ignored in saving process
        """
        group.attrs['class'] = 'DistributionSum'
        group.attrs['num_distributions'] = self.num_distributions
        subgroup = group.create_group('distributions')
        for (idistribution, distribution) in enumerate(self.distributions):
            distribution.fill_hdf5_group(\
                subgroup.create_group('{:d}'.format(idistribution)))
        create_hdf5_dataset(group, 'weights', data=self.weights)
        if save_metadata:
            self.save_metadata(group)
    
    @staticmethod
    def load_from_hdf5_group(group, *distribution_classes):
        """
        Loads a `DistributionSum` from the given hdf5 file group.
        
        Parameters
        ----------
        group : h5py.Group
            the same hdf5 file group which fill_hdf5_group was called on when
            this Distribution was saved
        
        Returns
        -------
        distribution : `DistributionSum`
            distribution created from the information in the given group
        """
        try:
            assert(group.attrs['class'] == 'DistributionSum')
            assert(\
                group.attrs['num_distributions'] == len(distribution_classes))
        except:
            raise ValueError("The given group does not appear to contain a " +\
                "DistributionSum object.")
        metadata = Distribution.load_metadata(group)
        weights = get_hdf5_value(group['weights'])
        subgroup = group['distributions']
        distributions = []
        for (icls, cls) in enumerate(distribution_classes):
            subsubgroup = subgroup['{:d}'.format(icls)]
            distribution = cls.load_from_hdf5_group(subsubgroup)
            distributions.append(distribution)
        return DistributionSum(distributions, weights, metadata=metadata)
    
    @property
    def can_give_confidence_intervals(self):
        """
        This distribution cannot give confidence intervals because its
        cumulative distribution function cannot be inverted in general.
        """
        return False

