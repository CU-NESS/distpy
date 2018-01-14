"""
File: JumpingDistributionSampler.py
Author: Keith Tauscher
Date: 26 Dec 2017

Description: File containing class implementing the abstract Sampler class from
             emcee using distpy's JumpingDistributionSet objects.
"""
import numpy as np
from emcee import Sampler as emceeSampler
from .JumpingDistributionSet import JumpingDistributionSet

try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str

class MetropolisHastingsSampler(emceeSampler):
    """
    """
    def __init__(self, parameters, nwalkers, logprobability,\
        jumping_distribution_set, args=[], kwargs={}):
        """
        """
        self.parameters = parameters
        self.nwalkers = nwalkers
        super(MetropolisHastingsSampler, self).__init__(len(self.parameters),\
            logprobability, *args, **kwargs)
        self.jumping_distribution_set = jumping_distribution_set
    
    @property
    def nwalkers(self):
        """
        """
        if not hasattr(self, '_nwalkers'):
            raise AttributeError("nwalkers referenced before it was set.")
        return self._nwalkers
    
    @nwalkers.setter
    def nwalkers(self, value):
        """
        """
        if isinstance(value, int):
            self._nwalkers = value
        else:
            raise TypeError("nwalkers was set to a non-int.")
    
    @property
    def parameters(self):
        """
        """
        if not hasattr(self, '_parameters'):
            raise AttributeError("parameters referenced before it was set.")
        return self._parameters
    
    @parameters.setter
    def parameters(self, value):
        """
        """
        if type(value) in [list, tuple, np.ndarray]:
            if all([isinstance(element, basestring) for element in value]):
                self._parameters = [element for element in value]
            else:
                raise TypeError("At least one parameter given was not a " +\
                    "string.")
        else:
            raise TypeError("parameters was set to a non-sequence.")
    
    @property
    def num_parameters(self):
        """
        """
        if not hasattr(self, '_num_parameters'):
            self._num_parameters = len(self.parameters)
        return self._num_parameters
    
    @property
    def jumping_distribution_set(self):
        """
        """
        if not hasattr(self, '_jumping_distribution_set'):
            raise AttributeError("jumping_distribution_set referenced " +\
                "before it was set.")
        return self._jumping_distribution_set
    
    @jumping_distribution_set.setter
    def jumping_distribution_set(self, value):
        """
        """
        if isinstance(value, JumpingDistributionSet):
            if set(self.parameters) == set(value.params):
                self._jumping_distribution_set = value
            else:
                raise ValueError("The given jumping_distribution_set did " +\
                    "not contain exactly the same parameters as the given " +\
                    "list of parameters given to the initializer.")
        else:
            raise TypeError("jumping_distribution_set was set to something " +\
                "which is not a JumpingDistributionSet object.")
    
    def array_from_dictionary(self, dictionary):
        """
        """
        return\
            np.array([dictionary[parameter] for parameter in self.parameters])
    
    def dictionary_from_array(self, array):
        """
        """
        return {parameter: array[iparameter]\
            for (iparameter, parameter) in enumerate(self.parameters)}

    def reset(self):
        """
        Clear ``chain``, ``lnprobability`` and the bookkeeping parameters.

        """
        self._chain = np.empty((self.nwalkers, 0, self.dim))
        self._lnprob = np.empty((self.nwalkers, 0))
        self.iterations = 0
        self.naccepted = np.zeros(self.nwalkers)
        self._last_run_mcmc_result = None
    
    def sample(self, point, lnprob=None, randomstate=None, thin=1,\
        storechain=True, iterations=1):
        """
        Advances the chain ``iterations`` steps as an iterator

        :param point:
            The initial position vector.

        :param lnprob: (optional)
            The log posterior probability at position ``point``. If ``lnprob``
            is not provided, the initial value is calculated.

        :param rstate0: (optional)
            The state of the random number generator. See the
            :func:`random_state` property for details.

        :param iterations: (optional)
            The number of steps to run.

        :param thin: (optional)
            If you only want to store and yield every ``thin`` samples in the
            chain, set thin to an integer greater than 1.

        :param storechain: (optional)
            By default, the sampler stores (in memory) the positions and
            log-probabilities of the samples in the chain. If you are
            using another method to store the samples to a file or if you
            don't need to analyse the samples after the fact (for burn-in
            for example) set ``storechain`` to ``False``.

        At each iteration, this generator yields:

        * ``pos`` - The current positions of the chain in the parameter
          space.

        * ``lnprob`` - The value of the log posterior at ``pos`` .

        * ``rstate`` - The current state of the random number generator.

        """
        self.random_state = randomstate
        point = np.array(point)
        if lnprob is None:
            lnprob = np.array([self.get_lnprob(point[iwalker])\
                for iwalker in range(self.nwalkers)])
        # Resize the chain in advance.
        if storechain:
            nlinks = int(iterations / thin)
            chain_shape = (self.nwalkers, nlinks, self.num_parameters)
            self._chain =\
                np.concatenate((self._chain, np.zeros(chain_shape)), axis=1)
            likelihood_shape = (self.nwalkers, nlinks)
            self._lnprob = np.concatenate((self._lnprob,\
                np.zeros(likelihood_shape)), axis=1)
        i0 = self.iterations
        for i in range(int(iterations)):
            self.iterations += 1
            for iwalker in range(self.nwalkers):
                # Calculate the proposal distribution.
                source_dict = self.dictionary_from_array(point[iwalker])
                destination_dict =\
                    self.jumping_distribution_set.draw(source_dict)
                destination = self.array_from_dictionary(destination_dict)
                newlnprob = self.get_lnprob(destination)
                log_value_difference =\
                    self.jumping_distribution_set.log_value_difference(\
                    source_dict, destination_dict)
                diff = newlnprob - lnprob[iwalker] -\
                    self.jumping_distribution_set.log_value_difference(\
                    source_dict, destination_dict)
                # M-H acceptance ratio
                if diff < 0:
                    diff = np.exp(diff) - self._random.rand()
                if diff > 0:
                    point[iwalker,:] = destination
                    lnprob[iwalker] = newlnprob
                    self.naccepted[iwalker] += 1
            if storechain and (i % thin) == 0:
                ind = i0 + int(i / thin)
                self._chain[:,ind,:] = point
                self._lnprob[:,ind] = lnprob
            yield point, lnprob, self.random_state

