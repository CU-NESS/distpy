"""
File: distpy/distribution/DistributionHarmonizer.py
Author: Keith Tauscher
Date: 28 Jun 2018

Description: File containing class created to generate a set of
             DeterministicDistribution objects which harmonize N-1 known
             (assumed) distributions with 1 unknown (solved for with a
             user-defined solver once other components subtracted out)
             distribution.
"""
import numpy as np
from ..util import int_types
from ..transform import TransformList, NullTransform
from .DeterministicDistribution import DeterministicDistribution
from .DistributionSet import DistributionSet

class DistributionHarmonizer(object):
    """
    Class created to generate a set of DeterministicDistribution objects which
    harmonize N-1 known (assumed) distributions with 1 unknown (solved for with
    a user-defined solver once other components subtracted out) distribution.
    """
    def __init__(self, known_distribution_set, remaining_parameter_solver,\
        ndraw, **transforms):
        """
        Initializes a new DistributionHarmonizer with the given known (or
        assumed) distribution sets, a solver for parameters not included in the
        distribution sets, and the desired number of samples.
        
        known_distribution_set: DistributionSet object or sequence of
                                DistributionSet objects which describe the
                                parameters whose distribution is known
                                (and/or) assumed. If None, then no parameters
                                are assumed known
        remaining_parameter_solver: a Callable which returns a dictionary of
                                    solved-for parameters when a dictionary
                                    sample of known- (or assumed-) distribution
                                    parameters is given
        ndraw: a positive integer denoting the maximum number of desired
               samples
        """
        self.known_distribution_set = known_distribution_set
        self.remaining_parameter_solver = remaining_parameter_solver
        self.ndraw = ndraw
        self.transforms = transforms
    
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
    def known_distribution_set(self):
        """
        Property storing the distribution set describing the parameter whose
        distributions are known.
        """
        if not hasattr(self, '_known_distribution_set'):
            raise AttributeError("known_distribution_set was referenced " +\
                "before it was set.")
        return self._known_distribution_set
    
    @known_distribution_set.setter
    def known_distribution_set(self, value):
        """
        Setter for the distribution set(s) which are known (or assumed).
        
        value: DistributionSet object or sequence of DistributionSet objects
               which decribe the parameters whose distribution is known
               (and/or) assumed
        """
        if type(value) is type(None):
            self._known_distribution_set = DistributionSet()
        elif isinstance(value, DistributionSet):
            self._known_distribution_set = value
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
    def remaining_parameter_solver(self):
        """
        Property storing the solver Callable which returns a dictionary of
        solved-for parameters when a dictionary sample of known- (or assumed-)
        distribution parameters is given.
        """
        if not hasattr(self, '_remaining_parameter_solver'):
            raise AttributeError("remaining_parameter_solver was " +\
                "referenced before it was set.")
        return self._remaining_parameter_solver
    
    @remaining_parameter_solver.setter
    def remaining_parameter_solver(self, value):
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
        self._remaining_parameter_solver = value
    
    @property
    def ndraw(self):
        """
        Property storing the maximum number (an integer) of samples the
        DistributionSet returned by this class can yield.
        """
        if not hasattr(self, '_ndraw'):
            raise AttributeError("ndraw was referenced before it was set.")
        return self._ndraw
    
    @ndraw.setter
    def ndraw(self, value):
        """
        Setter for the maximum number of samples the DistributionSet returned
        by this class can yield.
        
        value: a positive integer
        """
        if type(value) in int_types:
            if value > 0:
                self._ndraw = value
            else:
                raise ValueError("ndraw was set to a non-positive integer.")
        else:
            raise TypeError("ndraw was set to a non-integer.")
    
    @property
    def full_distribution_set(self):
        """
        Property storing the full DistributionSet describing all parameters,
        known and unknown.
        """
        if not hasattr(self, '_full_distribution_set'):
            known_draw = self.known_distribution_set.draw(self.ndraw)
            known_parameter_names = self.known_distribution_set.params
            if known_parameter_names:
                known_sample = np.stack([known_draw[parameter]\
                    for parameter in known_parameter_names], axis=-1)
            else:
                known_sample = np.zeros((self.ndraw, 0))
            for idraw in range(self.ndraw):
                known_parameters = {parameter: known_draw[parameter][idraw]\
                    for parameter in known_draw}
                these_solved_for_parameters =\
                    self.remaining_parameter_solver(known_parameters)
                if idraw == 0:
                    solved_for_parameters =\
                        {parameter: [these_solved_for_parameters[parameter]]\
                        for parameter in these_solved_for_parameters}
                else:
                    for parameter in solved_for_parameters:
                        solved_for_parameters[parameter].append(\
                            these_solved_for_parameters[parameter])
            solved_for_parameters =\
                {parameter: np.array(solved_for_parameters[parameter])\
                for parameter in solved_for_parameters}
            unknown_parameter_names =\
                [parameter for parameter in solved_for_parameters]
            solved_for_sample = np.stack([solved_for_parameters[parameter]\
                for parameter in unknown_parameter_names], axis=-1)
            full_sample =\
                np.concatenate([known_sample, solved_for_sample], axis=-1)
            full_parameter_names =\
                known_parameter_names + unknown_parameter_names
            transform_list = []
            for name in full_parameter_names:
                if name in self.transforms:
                    transform_list.append(self.transforms[name])
                else:
                    transform_list.append(NullTransform())
            transform_list = TransformList(*transform_list)
            full_transformed_sample = transform_list(full_sample)
            full_distribution =\
                DeterministicDistribution(full_transformed_sample)
            self._full_distribution_set = DistributionSet([(\
                full_distribution, full_parameter_names, transform_list)])
        return self._full_distribution_set

