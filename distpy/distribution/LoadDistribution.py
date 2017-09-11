from .UniformDistribution import UniformDistribution
from .GammaDistribution import GammaDistribution
from .BetaDistribution import BetaDistribution
from .PoissonDistribution import PoissonDistribution
from .GeometricDistribution import GeometricDistribution
from .BinomialDistribution import BinomialDistribution
from .ExponentialDistribution import ExponentialDistribution
from .DoubleSidedExponentialDistribution import\
    DoubleSidedExponentialDistribution
from .WeibullDistribution import WeibullDistribution
from .EllipticalUniformDistribution import EllipticalUniformDistribution
from .TruncatedGaussianDistribution import TruncatedGaussianDistribution
from .GaussianDistribution import GaussianDistribution
from .ParallelepipedDistribution import ParallelepipedDistribution
from .LinkedDistribution import LinkedDistribution
from .SequentialDistribution import SequentialDistribution
from .GriddedDistribution import GriddedDistribution
from .UniformDirectionDistribution import UniformDirectionDistribution
from .GaussianDirectionDistribution import GaussianDirectionDistribution

try:
    import h5py
except:
    have_h5py = False
    no_h5py_error = NotImplementedError("Loading couldn't be completed " +\
                                        "because h5py couldn't be imported.")
else:
    have_h5py = True

def load_distribution_from_hdf5_group(group):
    """
    Loads a distribution from the given hdf5 group.
    
    group: the hdf5 file group from which to load the distribution
    
    returns: Distribution object of the correct type
    """
    try:
        class_name = group.attrs['class']
    except KeyError:
        raise ValueError("group given does not appear to contain a " +\
                         "distribution.")
    if class_name == 'GammaDistribution':
        shape = group.attrs['shape']
        scale = group.attrs['scale']
        return GammaDistribution(shape, scale=scale)
    elif class_name == 'BetaDistribution':
        alpha = group.attrs['alpha']
        beta = group.attrs['beta']
        return BetaDistribution(alpha, beta)
    elif class_name == 'PoissonDistribution':
        scale = group.attrs['scale']
        return PoissonDistribution(scale)
    elif class_name == 'GeometricDistribution':
        common_ratio = group.attrs['common_ratio']
        return GeometricDistribution(common_ratio)
    elif class_name == 'BinomialDistribution':
        probability_of_success = group.attrs['probability_of_success']
        number_of_trials = group.attrs['number_of_trials']
        return BinomialDistribution(probability_of_success, number_of_trials)
    elif class_name == 'ExponentialDistribution':
        rate = group.attrs['rate']
        shift = group.attrs['shift']
        return ExponentialDistribution(rate, shift=shift)
    elif class_name == 'DoubleSidedExponentialDistribution':
        mean = group.attrs['mean']
        variance = group.attrs['variance']
        return DoubleSidedExponentialDistribution(mean, variance)
    elif class_name == 'EllipticalUniformDistribution':
        mean = group['mean'].value
        covariance = group['covariance'].value
        return EllipticalUniformDistribution(mean, covariance)
    elif class_name == 'UniformDistribution':
        low = group.attrs['low']
        high = group.attrs['high']
        return UniformDistribution(low=low, high=high)
    elif class_name == 'WeibullDistribution': 
        shape = group.attrs['shape']
        scale = group.attrs['scale']
        return WeibullDistribution(shape=shape, scale=scale)
    elif class_name == 'TruncatedGaussianDistribution':
        mean = group.attrs['mean']
        variance = group.attrs['variance']
        low = group.attrs['low']
        high = group.attrs['high']
        return\
            TruncatedGaussianDistribution(mean, variance, low=low, high=high)
    elif class_name == 'GaussianDistribution':
        mean = group['mean'].value
        covariance = group['covariance'].value
        return GaussianDistribution(mean, covariance)
    elif class_name == 'ParallelepipedDistribution':
        center = group['center'].value
        face_directions = group['face_directions'].value
        distances = group['distances'].value
        return ParallelepipedDistribution(center, face_directions, distances)
    elif class_name == 'LinkedDistribution':
        numparams = group.attrs['numparams']
        shared_distribution =\
            load_distribution_from_hdf5_group(group['shared_distribution'])
        return LinkedDistribution(shared_distribution, numparams)
    elif class_name == 'SequentialDistribution':
        numparams = group.attrs['numparams']
        shared_distribution =\
            load_distribution_from_hdf5_group(group['shared_distribution'])
        return SequentialDistribution(shared_distribution=shared_distribution,\
            numpars=numparams)
    elif class_name == 'GriddedDistribution':
        variables = []
        ivar = 0
        while ('variable_%i' % (ivar,)) in group.attrs:
            variables.append(group.attrs['variable_%i' % (ivar,)])
            ivar += 1
        pdf = group['pdf'].value
        return GriddedDistribution(variables=variables, pdf=pdf)
    elif class_name == 'UniformDirectionDistribution':
        pointing_center = tuple(group.attrs['pointing_center'])
        low_theta = group.attrs['low_theta']
        high_theta = group.attrs['high_theta']
        low_phi = group.attrs['low_phi']
        high_phi = group.attrs['high_phi']
        return UniformDirectionDistribution(low_theta=low_theta,\
            high_theta=high_theta, low_phi=low_phi, high_phi=high_phi,\
            pointing_center=pointing_center, psi_center=psi_center)
    elif class_name == 'GaussianDirectionDistribution':
        pointing_center = tuple(group.attrs['pointing_center'])
        sigma = group.attrs['sigma']
        return GaussianDirectionDistribution(pointing_center=pointing_center,\
            sigma=sigma, degrees=False)
    else:
        raise ValueError("The class of the Distribution was not recognized.")

def load_distribution_from_hdf5_file(file_name):
    """
    Loads Distribution object of any subclass from an hdf5 file at the given
    file name.
    
    file_name: location of hdf5 file containing distribution
    
    returns: Distribution object contained in the hdf5 file
    """
    if have_h5py:
        hdf5_file = h5py.File(file_name, 'r')
        distribution = load_distribution_from_hdf5_group(hdf5_file)
        hdf5_file.close()
        return distribution
    else:
        raise no_h5py_error
