"""
File: examples/jumping/metropolis_hastings_sampler.py
Author: Keith Tauscher
Date: 12 Mar 2019

Description: Example script showing how to use the MetropolisHastingsSampler to
             sample a simple distribution (in this case the lnprob function is
             a simple negative parabola, implying a Gaussian likelihood). This
             script also allows for easy testing of the paralellization, which,
             on a given system, is efficient once the logprobability function
             is slower than some characteristic time. On my system, 4 threads
             become faster than one when the logprobability takes about 1 ms to
             compute.
"""
from __future__ import division
import time, sys
import numpy as np
from distpy import GaussianDistribution, DistributionSet,\
    GaussianJumpingDistribution, JumpingDistributionSet,\
    MetropolisHastingsSampler

if len(sys.argv) == 1:
    print("Using nthreads=1 because none was given. To give one, call this " +\
        "script with `python metropolis_hastings_sampler.py $NTHREADS`")
    nthreads = 1
else:
    nthreads = int(sys.argv[1])

parameters = ['x', 'y', 'z']
mean = np.array([1, 2, 4])
nwalkers = 4
numsteps = 10
rstate = None
wait_time = 0.1

def logprobability(pars):
    time.sleep(wait_time)
    return np.sum(np.power(pars - mean, 2)) / (-2)
covariance = np.identity(len(parameters))
guess_distribution_set = DistributionSet([\
    (GaussianDistribution(mean, covariance), parameters, None)])
jumping_distribution_set = JumpingDistributionSet([\
    (GaussianJumpingDistribution(covariance), parameters, None)])

sampler = MetropolisHastingsSampler(parameters, nwalkers, logprobability,\
    jumping_distribution_set, nthreads=nthreads)

guesses = guess_distribution_set.draw(nwalkers)
guesses = np.stack([guesses[parameter] for parameter in parameters], axis=1)
lnprob = np.array(list(map(logprobability, guesses)))

start_time = time.time()
sampler.run_mcmc(guesses, numsteps, rstate0=rstate, lnprob0=lnprob)
end_time = time.time()
duration = end_time - start_time

print("(nwalkers, ncheckpoints, nparameters)={}".format(\
    (nwalkers, numsteps, len(parameters))))
print("sampler.chain.shape={}".format(sampler.chain.shape))
print("sampler.lnprobability.shape={}".format(sampler.lnprobability.shape))
print("The sampling took {:.6f} s.".format(duration))

