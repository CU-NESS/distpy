"""
File: distpy/distribution/DistributionHarmonizer.py
Author: Keith Tauscher
Date: 27 May 2019

Description: File containing class created to generate a set of
             DeterministicDistribution objects which harmonize N-M known
             (assumed) marginal distributions with M conditional (solved for
             with a user-defined solver once other components subtracted out)
             distributions.
"""
import numpy as np
from ..util import int_types
from ..transform import TransformList, NullTransform
from .DeterministicDistribution import DeterministicDistribution
from .DistributionSet import DistributionSet

class DistributionHarmonizer(object):
    """
    Class created to generate a set of DeterministicDistribution objects which
    harmonize N-M known (assumed) marginal distributions with M conditional
    (solved for with a user-defined solver once other components subtracted
    out) distributions.
    """
    def __init__(self, marginal_distribution_set, conditional_solver,\
        marginal_draws, conditional_draws=None, **transforms):
        """
        Initializes a new DistributionHarmonizer with the given known (or
        assumed) distribution sets, a solver for parameters not included in the
        distribution sets, and the desired number of samples.
        
        marginal_distribution_set: DistributionSet object or sequence of
                                   DistributionSet objects which describe the
                                   parameters whose distribution is known
                                   (and/or) assumed. If None, then no
                                   parameters are assumed known
        conditional_solver: a Callable which returns:
                            1) if conditional_draws is None, a dictionary of
                               solved-for parameters when a dictionary sample
                               of known- (or assumed-) distribution parameters
                               is given (if transforms are given,
                               conditional_solver should return parameters in
                               untransformed space)
                            2) if conditional_draws is not None, a
                               DistributionSet object that represents the
                               conditional distribution of the unknown
                               parameters given the known parameters (if
                               transforms are given, the DistributionSet
                               returned by conditional_solver should include
                               these transforms in order to return values in
                               untransformed space)
        marginal_draws: a positive integer denoting the number of times the
                        known distributions are drawn from
        conditional_draws: if None (default), then the conditional distribution
                                              is effectively degenerate, with
                                              only the maximum probability
                                              value being drawn (see case 1 in
                                              the documentation of the
                                              conditional_solver argument
                                              above)
                           otherwise, then this should be a positive integer
                                      determining the number of times the
                                      conditional distribution returned by
                                      conditional_solver (see case 2 in the
                                      documentation of the conditional_solver
                                      argument above) should be drawn from for
                                      each of the marginal_draws number of
                                      draws from the marginal distribution.
        transforms: dictionary of transform objects defining the space in which
                    the DeterministicDistribution object included in the
                    joint_distribution_set property should exist
        """
        self.marginal_distribution_set = marginal_distribution_set
        self.conditional_solver = conditional_solver
        self.marginal_draws = marginal_draws
        self.conditional_draws = conditional_draws
        self.transforms = transforms
    
    @property
    def conditional_draws(self):
        """
        Property storing the number of draws to take from the conditional
        distribution for each draw of the marginal distribution. This must be
        either None or a positive integer. If it is None, conditional_solver
        property must return a dictionary of unknown parameters when passed a
        dictionary of known parameters. If it is a positive integer,
        conditional_solver property must return a DistributionSet object
        representing the conditional distribution when passed a dictionary of
        parameters drawn from the marginal distribution is passed.
        """
        if not hasattr(self, '_conditional_draws'):
            raise AttributeError("conditional_draws was referenced before " +\
                "it was set.")
        return self._conditional_draws
    
    @conditional_draws.setter
    def conditional_draws(self, value):
        """
        Setter for the conditional draws property.
        
        value: either None or a positive integer. The returned value of
               conditional_solver depends on whether this is None or an
               integer. See the documentation on this property's getter for
               more information.
        """
        if type(value) is type(None):
            self._conditional_draws = value
        elif type(value) in int_types:
            if value > 0:
                self._conditional_draws = value
            else:
                raise ValueError("conditional_draws was set to a " +\
                    "non-positive integer.")
        else:
            raise TypeError("conditional_draws was set to neither None nor " +\
                "an integer.")
    
    @property
    def transforms(self):
        """
        Property storing the transform to use for each parameter name (in a
        dictionary).
        """
        if not hasattr(self, '_transforms'):
            raise AttributeError("transforms was referenced before it was " +\
                "set.")
        return self._transforms
    
    @transforms.setter
    def transforms(self, value):
        """
        Setter for the transforms to apply to the parameters.
        
        value: a dictionary with parameter names as keys
        """
        if isinstance(value, dict):
            self._transforms = value
        else:
            raise TypeError("transforms was set to a non-dictionary.")
    
    @property
    def marginal_distribution_set(self):
        """
        Property storing the distribution set describing the parameter whose
        marginal distribution is known.
        """
        if not hasattr(self, '_marginal_distribution_set'):
            raise AttributeError("marginal_distribution_set was referenced " +\
                "before it was set.")
        return self._marginal_distribution_set
    
    @marginal_distribution_set.setter
    def marginal_distribution_set(self, value):
        """
        Setter for the marginal distribution set(s) for the known (or assumed)
        parameters.
        
        value: DistributionSet object or sequence of DistributionSet objects
               which decribe the parameters whose distribution is known
               (and/or) assumed
        """
        if type(value) is type(None):
            self._marginal_distribution_set = DistributionSet()
        elif isinstance(value, DistributionSet):
            self._marginal_distribution_set = value
        elif type(value) in sequence_types:
            if all([isinstance(element, DistributionSet)\
                for element in value]):
                self._distribution_sets = sum(value, DistributionSet())
            else:
                raise TypeError("Not all elements of the sequence to which " +\
                    "distribution_sets was set were DistributionSet objects.")
        else:
            raise TypeError("distribution_sets was set to neither a " +\
                "DistributionSet object nor a sequence of DistributionSet " +\
                "objects.")
    
    @property
    def conditional_solver(self):
        """
        Property storing the solver Callable which returns a dictionary of
        solved-for parameters when a dictionary sample of known- (or assumed-)
        distribution parameters is given.
        """
        if not hasattr(self, '_conditional_solver'):
            raise AttributeError("conditional_solver was referenced before " +\
                "it was set.")
        return self._conditional_solver
    
    @conditional_solver.setter
    def conditional_solver(self, value):
        """
        Property storing the solver which computes values of parameters whose
        distributions are unknown (or not assumed) from the values of
        parameters whose distributions are known (or assumed). Usually, this
        involves a sort of conditionalization (e.g. over a likelihood
        function).
        
        value: a Callable which returns a dictionary of solved-for parameters
               when a dictionary sample of known- (or assumed-) distribution
               parameters is given
        """
        self._conditional_solver = value
    
    @property
    def marginal_draws(self):
        """
        Property storing the maximum number (an integer) of samples the
        DistributionSet returned by this class can yield.
        """
        if not hasattr(self, '_marginal_draws'):
            raise AttributeError("marginal_draws was referenced before it " +\
                "was set.")
        return self._marginal_draws
    
    @marginal_draws.setter
    def marginal_draws(self, value):
        """
        Setter for the maximum number of samples the DistributionSet returned
        by this class can yield.
        
        value: a positive integer
        """
        if type(value) in int_types:
            if value > 0:
                self._marginal_draws = value
            else:
                raise ValueError("marginal_draws was set to a non-positive " +\
                    "integer.")
        else:
            raise TypeError("marginal_draws was set to a non-integer.")
    
    @property
    def joint_distribution_set(self):
        """
        Property storing the joint DistributionSet describing all parameters,
        known and unknown.
        """
        if not hasattr(self, '_joint_distribution_set'):
            marginal_draw =\
                self.marginal_distribution_set.draw(self.marginal_draws)
            marginal_parameter_names = self.marginal_distribution_set.params
            if marginal_parameter_names:
                marginal_sample = np.stack([marginal_draw[parameter]\
                    for parameter in marginal_parameter_names], axis=-1)
            else:
                marginal_sample = np.zeros((self.marginal_draws, 0))
            if type(self.conditional_draws) is not type(None):
                marginal_sample = np.reshape(marginal_sample[:,np.newaxis,:] *\
                    np.ones((1, self.conditional_draws, 1)),\
                    (-1, marginal_sample.shape[-1]))
            for imarginal_draw in range(self.marginal_draws):
                marginal_parameters =\
                    {parameter: marginal_draw[parameter][imarginal_draw]\
                    for parameter in marginal_draw}
                these_conditional_parameters =\
                    self.conditional_solver(marginal_parameters)
                if type(self.conditional_draws) is type(None):
                    these_conditional_parameters =\
                        {parameter: [these_conditional_parameters[parameter]]\
                        for parameter in these_conditional_parameters}
                else:
                    these_conditional_parameters =\
                        these_conditional_parameters.draw(\
                        self.conditional_draws)
                if imarginal_draw == 0:
                    conditional_parameters = {parameter:\
                        [these_conditional_parameters[parameter]]\
                        for parameter in these_conditional_parameters}
                else:
                    for parameter in conditional_parameters:
                        conditional_parameters[parameter].append(\
                            these_conditional_parameters[parameter])
            conditional_parameters =\
                {parameter: np.concatenate(conditional_parameters[parameter])\
                for parameter in conditional_parameters}
            conditional_parameter_names =\
                [parameter for parameter in conditional_parameters]
            conditional_sample = np.stack([conditional_parameters[parameter]\
                for parameter in conditional_parameter_names], axis=-1)
            joint_sample =\
                np.concatenate([marginal_sample, conditional_sample], axis=-1)
            joint_sample =\
                joint_sample[np.random.permutation(len(joint_sample)),:]
            joint_parameter_names =\
                marginal_parameter_names + conditional_parameter_names
            transform_list = []
            for name in joint_parameter_names:
                if name in self.transforms:
                    transform_list.append(self.transforms[name])
                else:
                    transform_list.append(NullTransform())
            transform_list = TransformList(*transform_list)
            joint_transformed_sample = transform_list(joint_sample)
            joint_distribution =\
                DeterministicDistribution(joint_transformed_sample)
            self._joint_distribution_set = DistributionSet([(\
                joint_distribution, joint_parameter_names, transform_list)])
        return self._joint_distribution_set

