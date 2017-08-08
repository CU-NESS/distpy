distpy is a clean and simple Python package meant to store analytical distributions efficiently and effectively. The module has no submodules, and in it are defined the distributions defined below.

The included univariate distributions (and their initializations, equals signs indicate optional parameters and their default values) are (pdf's apply only in support):


(1) GammaDistribution(shape, scale=1)

(pdf) --- f(x) = (x/scale)^(shape-1) * e^(-x/scale) / scale / Gamma(shape)

(support) --- x > 0

(mean) --- shape * scale

(variance) --- shape * scale^2


(2) ChiSquaredDistribution(dof)

(pdf) --- f(x) = ((1/2)^(dof/2) / gamma(dof/2)) * x^((dof/2)-1) * e^(-x/2)

(support) --- x > 0

(mean) --- dof

(variance) --- 2 * dof


(3) BetaDistribution(alpha, beta)

(pdf) --- f(x) = x^(alpha - 1) * (1 - x)^(beta - 1) / Beta(alpha, beta)

(support) --- 0 < x < 1

(mean) --- alpha / (alpha+beta)

(variance) (alpha * beta) / (alpha + beta)^2 / (alpha + beta + 1)


(4) PoissonDistribution(scale)

(pdf) ---  f(k) = scale^k * e^(-scale) / k!

(support) --- non-negative integer k

(mean) --- scale

(variance) --- scale


(5) GeometricDistribution(common_ratio)

(pdf) --- f(k) = (1 - common_ratio) * common_ratio^k

(support) --- non-negative integer k

(mean) --- common_ratio / (1 - common_ratio)

(variance) --- common_ratio / (1 - common_ratio)^2


(6) BinomialDistribution(p, n)

(pdf) --- f(x) = (n choose k) p^k (1-p)^(n-k)

(support) --- integer k satisfying (0 <= k <= n)

(mean) --- n * p

(variance) --- n * p * (1-p)


(7) ExponentialDistribution(rate, shift=0)

(pdf) --- f(x) = rate * e^(-rate * (x - shift))

(support) --- x>0

(mean) --- 1 / rate

(variance) --- 1 / rate^2


(8) DoubleSidedExponentialDistribution(mean, variance)

(pdf) --- f(x) = e^(-|(x - mean) / sqrt(variance/2)|) / (sqrt(2*variance))

(support) --- real x

(mean) --- mean

(variance) --- variance


(9) UniformDistribution(low, high)

(pdf) --- f(x) = 1 / (high - low)

(support) --- low < x < high

(mean) --- (low + high) / 2

(variance) --- (high - low)^2 / 12


(10) GaussianDistribution(mean, variance)

(pdf) --- f(x) = e^(- (x - mean)^2 / (2 * variance)) / sqrt(2pi * variance)

(support) --- -infty < x < infty

(mean) --- mean

(variance) --- variance


(11) TruncatedGaussianDistribution(mean, variance, low, high)

(pdf) --- rescaled and truncated version of pdf of GaussianDistribution

(support) --- low < x < high

(mean) --- no convenient expression; in limit, approaches mean

(variance) --- no convenient expression; in limit, approaches variance


(12) WeibullDistribution(shape=1, scale=1.)

(pdf) --- f(x) = (shape / scale) * (x / scale)^(shape - 1) * e^(-(x / scale)^shape)

(support) --- x > 0

(mean) --- scale * Gamma(1 + (1 / shape))

(variance) --- scale^2 * [Gamma(1 + (2 / shape)) - (Gamma(1 + (1 / shape)))^2]



And the following multivariate distributions are included:


(1) GaussianDistribution(mean, covariance)

(pdf) --- f(x) = e^(- (x-mean)^T covariance^(-1) (x-mean) / 2) / np.sqrt( (2 * pi)^numparams * det(covariance))

(support) --- x is vector in R^N where N is number of dimensions of mean

(mean) --- mean

(variance) --- covariance


(2) ParallelepipedDistribution(center, face_directions, distances, norm_dirs=True)

(pdf) --- f(x) = 1 / det(matrix_of_directions_from_vertex)

(support) --- in the parallelogram described by | (x - center) dot face_directions[i] | <= distances[i] ; If norm_dirs=False, face_directions can have mag != 1

(mean) --- center

(variance) --- no nice expression; depends on angles between face directions


(3) LinkedDistribution(shared_distribution, numparams)

(pdf) --- f(x1, x2, x3, ...) = shared_distribution(x1) * prod_{i=2}^N delta(xi-x1)

(support) --- x1=x2=x3=...=xN where all are in support of shared_distribution

(mean) --- (mu, mu, mu, ...) where mu is mean of shared_distribution

(variance) --- no convenient expression


(4) SequentialDistribution(shared_distribution, numparams)

(pdf) --- f(x1, x2, x3, ...) = prod_{i=1}^N shared_distribution(xi) * is_sorted(x1,x2,...)

(support) --- x1<x2<x3<...<xN where all are in support of shared_distribution

(mean) --- no convenient expression

(variance) --- no convenient expression


(5) GriddedDistribution(variables, pdf=None)

(pdf) --- user-defined through ndarray, if pdf=None, assumed uniform

(support) --- must be rectangular; defined through variable ranges in variables

(mean) --- unknown a priori

(variance) --- unknown a priori


(6) EllipticalUniformDistribution(mean, cov)

(pdf) --- f(X)=1 when X is inside (X-mean)^T cov^-1 (X-mean)<= N+2

(support) --- hyperellipsoid defined above

(mean) --- mean

(variance) --- cov

