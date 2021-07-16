"""
Module containing class representing a jumping distribution given by a weighted
sum of other jumping distributions. If the PDFs of the constituent jumping
distributions are \\(g_1,g_2,\\ldots,g_N\\), this distribution's PDF is given
by $$f(\\boldsymbol{x},\\boldsymbol{y})=\\frac{\
\\sum_{k=1}^Nw_kg_k(\\boldsymbol{x},\\boldsymbol{y})}{\\sum_{k=1}^Nw_k}$$

**File**: $DISTPY/distpy/jumping/JumpingDistributionSum.py  
**Author**: Keith Tauscher  
**Date**: 3 Jul 2021
"""
import numpy as np
from ..util import int_types, sequence_types, create_hdf5_dataset,\
    get_hdf5_value
from .JumpingDistribution import JumpingDistribution

rand = np.random

class JumpingDistributionSum(JumpingDistribution):
    """
    Class representing a jumping distribution given by a weighted sum of other
    jumping distributions. If the PDFs of the constituent jumping distributions
    are \\(g_1,g_2,\\ldots,g_N\\), this distribution's PDF is given by
    $$f(\\boldsymbol{x},\\boldsymbol{y})=\\frac{\
    \\sum_{k=1}^Nw_kg_k(\\boldsymbol{x},\\boldsymbol{y})}{\\sum_{k=1}^Nw_k}$$
    """
    def __init__(self, jumping_distributions, weights):
        """
        Creates a new `JumpingDistributionSum` object out of the given
        `distpy.jumping.JumpingDistribution.JumpingDistribution` objects.
        
        Parameters
        ----------
        jumping_distributions : sequence
            sequence of
            `distpy.jumping.JumpingDistribution.JumpingDistribution` objects
            with the same
            `distpy.jumping.JumpingDistribution.JumpingDistribution.numparams`
            that will be weighted
        weights : sequence
            weighted with which to combine `jumping_distributions` (need not be
            normalized but they must all be positive)
        """
        self.jumping_distributions = jumping_distributions
        self.weights = weights
    
    @property
    def jumping_distributions(self):
        """
        A list of `distpy.jumping.JumpingDistribution.JumpingDistribution`
        objects which are weighted by this `JumpingDistributionSum`.
        """
        if not hasattr(self, '_jumping_distributions'):
            raise AttributeError("jumping_distributions was referenced " +\
                "before it was set.")
        return self._jumping_distributions
    
    @jumping_distributions.setter
    def jumping_distributions(self, value):
        """
        Setter for the `JumpingDistributionSum.jumping_distributions`.
        
        Parameters
        ----------
        value : sequence
            sequence of
            `distpy.jumping.JumpingDistribution.JumpingDistribution` objects
            with the same
            `distpy.jumping.JumpingDistribution.JumpingDistribution.numparams`
            that will be weighted
        """
        if type(value) in sequence_types:
            if all([isinstance(element, JumpingDistribution)\
                for element in value]):
                unique_nums_of_parameters =\
                    np.unique([element.numparams for element in value])
                unique_is_discretes =\
                    np.unique([element.is_discrete for element in value])
                if (len(unique_nums_of_parameters) == 1) and\
                    (len(unique_is_discretes) == 1):
                    self._jumping_distributions =\
                        [element for element in value]
                    self._numparams = unique_nums_of_parameters[0]
                    self._is_discrete = unique_is_discretes[0]
                else:
                    raise ValueError("Not all jumping_distributions added " +\
                        "together here had the same number of parameters.")
            else:
                raise TypeError("At least one element of the " +\
                    "jumping_distributions sequence was not a " +\
                    "JumpingDistribution object.")
        else:
            raise TypeError("jumping_distributions was set to a non-sequence.")
    
    @property
    def num_jumping_distributions(self):
        """
        The number of `distpy.jumping.JumpingDistribution.JumpingDistribution`
        objects which make up this `JumpingDistributionSum` object.
        """
        if not hasattr(self, '_num_jumping_distributions'):
            self._num_jumping_distributions = len(self.jumping_distributions)
        return self._num_jumping_distributions
    
    @property
    def weights(self):
        """
        The weights with which to combine the distributions.
        """
        if not hasattr(self, '_weights'):
            raise AttributeError("weights was referenced before it was set.")
        return self._weights
    
    @weights.setter
    def weights(self, value):
        """
        Setter for `JumpingDistributionSum.weights`.
        
        Parameters
        ----------
        weights : sequence
            weighted with which to combine `jumping_distributions` (need not be
            normalized but they must all be positive)
        """
        if type(value) in sequence_types:
            if len(value) == self.num_jumping_distributions:
                value = np.array(value)
                if np.all(np.zeros(self.num_jumping_distributions) < value):
                    self._weights = value
                else:
                    raise ValueError("At least one weight was non-positive.")
            else:
                raise ValueError("The length of the weights sequence was " +\
                    "not the same as the length of the " +\
                    "jumping_distributions sequence.")
        else:
            raise TypeError("weights was set to a non-sequence.")
    
    @property
    def total_weight(self):
        """
        The sum of all of the weights given to this distribution.
        """
        if not hasattr(self, '_total_weight'):
            self._total_weight = np.sum(self.weights)
        return self._total_weight
    
    @property
    def discrete_cdf_values(self):
        """
        The upper bound of CDF values allocated to each model. The sequence is
        monotonically increasing and its last element is equal to 1.
        """
        if not hasattr(self, '_discrete_cdf_values'):
            self._discrete_cdf_values =\
                np.cumsum(self.weights) / self.total_weight
        return self._discrete_cdf_values
    
    def draw(self, source, shape=None, random=rand):
        """
        Draws destination point(s) from this jumping distribution given a
        source point.
        
        Parameters
        ----------
        source : number or numpy.ndarray
            - if this `JumpingDistributionSum` is univariate, source should be
            a single number
            - otherwise, source should be `numpy.ndarray` of shape (numparams,)
        shape : None or int or tuple
            - if None, a single destination is returned
                - if this distribution is univariate, a single number is
                returned
                - if this distribution is multivariate, a 1D `numpy.ndarray`
                describing the coordinates of the destination is returned
            - if int \\(n\\), \\(n\\) destinations are returned
                - if this distribution is univariate, a 1D `numpy.ndarray` of
                length \\(n\\) is returned
                - if this distribution describes \\(p\\) dimensions, a 2D
                `numpy.ndarray` is returned whose shape is \\((n,p)\\)
            - if tuple of ints \\((n_1,n_2,\\ldots,n_k)\\),
            \\(\\prod_{m=1}^kn_m\\) destinations are returned
                - if this distribution is univariate, a `numpy.ndarray` of
                shape \\((n_1,n_2,\\ldots,n_k)\\) is returned
                - if this distribution describes \\(p\\) parameters, a
                `numpy.ndarray` of shape \\((n_1,n_2,\\ldots,n_k,p)\\) is
                returned
        random : numpy.random.RandomState
            the random number generator to use (default: `numpy.random`)
        
        Returns
        -------
        destinations : number or numpy.ndarray
            either single value or array of values. See documentation on
            `shape` above for the type of the returned value
        """
        if type(shape) is type(None):
            uniform = random.uniform()
            idistribution = np.argmax(self.discrete_cdf_values > uniform)
            jumping_distribution = self.jumping_distributions[idistribution]
            return jumping_distribution.draw(source, random=random)
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
            for (idistribution, jumping_distribution) in\
                enumerate(self.jumping_distributions):
                cdf_lower_bound = cdf_upper_bound
                cdf_upper_bound = self.discrete_cdf_values[idistribution]
                relevant_indices = np.nonzero((uniforms < cdf_upper_bound) &\
                    (uniforms >= cdf_lower_bound))[0]
                ndraw = len(relevant_indices)
                draws[relevant_indices,...] = jumping_distribution.draw(\
                    source, shape=ndraw, random=random)
            return np.reshape(draws, final_shape)
    
    def log_value(self, source, destination):
        """
        Computes the log-PDF of jumping from `source` to `destination`.
        
        Parameters
        ----------
        source : number or numpy.ndarray
            - if this distribution is univariate, `source` must be a number
            - if this distribution describes \\(p\\) parameters, `source` must
            be a 1D `numpy.ndarray` of length \\(p\\)
        destination : number or numpy.ndarray
            - if this distribution is univariate, `destination` must be a
            number
            - if this distribution describes \\(p\\) parameters, `destination`
            must be a 1D `numpy.ndarray` of length \\(p\\)
        
        Returns
        -------
        log_pdf : float
            if the distribution is \\(f(\\boldsymbol{x},\\boldsymbol{y})=\
            \\text{Pr}[\\boldsymbol{y}|\\boldsymbol{x}]\\), `source` is
            \\(\\boldsymbol{x}\\) and `destination` is \\(\\boldsymbol{y}\\),
            then `log_pdf` is given by
            \\(\\ln{f(\\boldsymbol{x},\\boldsymbol{y})}\\)
        """
        values = np.exp([jumping_distribution.log_value(source, destination)\
            for jumping_distribution in self.jumping_distributions])
        return np.log(np.sum(values * self.weights) / self.total_weight)
    
    @property
    def numparams(self):
        """
        The integer number of parameters described by this distribution.
        """
        if not hasattr(self, '_numparams'):
            raise AttributeError("numparams was referenced before " +\
                "distributions was set.")
        return self._numparams
    
    def __eq__(self, other):
        """
        Tests for equality between this `JumpingDistributionSum` and `other`.
        
        Parameters
        ----------
        other : object
            object with which to check for equality
        
        Returns
        -------
        result : bool
            True if and only if object is a `JumpingDistributionSum` with the
            same distributions and weights.
        """
        if isinstance(other, JumpingDistributionSum):
            if self.num_jumping_distributions == other.num_jumping_distributions:
                distributions_equal = all([(sdistribution == odistribution)\
                    for (sdistribution, odistribution) in zip(\
                    self.jumping_distributions, other.jumping_distributions)])
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
        A boolean describing whether this distribution is discrete (True) or
        continuous (False).
        """
        if not hasattr(self, '_is_discrete'):
            raise AttributeError("is_discrete was referenced before " +\
                "jumping_distributions was set.")
        return self._is_discrete
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with information about this
        distribution.
        
        Parameters
        ----------
        group : h5py.Group
            hdf5 file group to fill with information about this distribution
        """
        group.attrs['class'] = 'JumpingDistributionSum'
        group.attrs['num_jumping_distributions'] =\
            self.num_jumping_distributions
        subgroup = group.create_group('jumping_distributions')
        for (idistribution, jumping_distribution) in\
            enumerate(self.jumping_distributions):
            jumping_distribution.fill_hdf5_group(\
                subgroup.create_group('{:d}'.format(idistribution)))
        create_hdf5_dataset(group, 'weights', data=self.weights)
    
    @staticmethod
    def load_from_hdf5_group(group, *jumping_distribution_classes):
        """
        Loads a `JumpingDistributionSum` from the given hdf5 file group.
        
        Parameters
        ----------
        group : h5py.Group
            the same hdf5 file group which fill_hdf5_group was called on when
            this `JumpingDistributionSum` was saved
        jumping_distribution_classes : sequence
            sequence (of length
            `JumpingDistributionSum.num_jumping_distributions`) of class
            objects which can be used to load the sub-distributions
        
        Returns
        -------
        loaded : `JumpingDistributionSum
            a `JumpingDistributionSum` object created from the information in
            the given group
        """
        try:
            assert(group.attrs['class'] == 'JumpingDistributionSum')
            assert(group.attrs['num_jumping_distributions'] ==\
                len(jumping_distribution_classes))
        except:
            raise ValueError("The given group does not appear to contain a " +\
                "JumpingDistributionSum object.")
        weights = get_hdf5_value(group['weights'])
        subgroup = group['jumping_distributions']
        jumping_distributions = []
        for (icls, cls) in enumerate(jumping_distribution_classes):
            subsubgroup = subgroup['{:d}'.format(icls)]
            jumping_distribution = cls.load_from_hdf5_group(subsubgroup)
            jumping_distributions.append(jumping_distribution)
        return JumpingDistributionSum(jumping_distributions, weights)
