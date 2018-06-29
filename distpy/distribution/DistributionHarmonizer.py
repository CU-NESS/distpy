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
from .DeterministicDistribution import DeterministicDistribution
from .DistributionSet import DistributionSet

class DistributionHarmonizer(object):
    """
    Class created to generate a set of DeterministicDistribution objects which
    harmonize N-1 known (assumed) distributions with 1 unknown (solved for with
    a user-defined solver once other components subtracted out) distribution.
    """
    def __init__(self, known_distribution_set, remaining_parameter_solver,\
        ndraw):
        """
        Initializes a new DistributionHarmonizer with the given known (or
        assumed) distribution sets, a solver for parameters not included in the
        distribution sets, and the desired number of samples.
        
        known_distribution_set: DistributionSet object or sequence of
                                DistributionSet objects which describe the
                                parameters whose distribution is known
                                (and/or) assumed
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
        if isinstance(value, DistributionSet):
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
            sampled_known_distribution_set = DistributionSet()
            for parameter in known_draw:
                distribution = DeterministicDistribution(known_draw[parameter])
                sampled_known_distribution_set.add_distribution(distribution,\
                    parameter)
            for idraw in range(self.ndraw):
                known_parameters = {parameter: known_draw[parameter][idraw]\
                    for parameter in known_draw}
                solved_for_parameters =\
                    self.remaining_parameter_solver(known_parameters)
                if idraw == 0:
                    full_solved_for_parameters =\
                        {parameter: [solved_for_parameters[parameter]]\
                        for parameter in solved_for_parameters}
                else:
                    for parameter in full_solved_for_parameters:
                        full_solved_for_parameters[parameter].append(\
                            solved_for_parameters[parameter])
            full_solved_for_parameters =\
                {parameter: np.array(full_solved_for_parameters[parameter])\
                for parameter in full_solved_for_parameters}
            solved_for_distribution_set = DistributionSet()
            for parameter in full_solved_for_parameters:
                distribution = DeterministicDistribution(\
                    full_solved_for_parameters[parameter])
                solved_for_distribution_set.add_distribution(distribution,\
                    parameter)
            self._full_distribution_set =\
                sampled_known_distribution_set + solved_for_distribution_set
        return self._full_distribution_set

