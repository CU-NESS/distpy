---------------
distpy overview
---------------

distpy is a clean and simple Python package meant to store analytical distributions efficiently and effectively. It has four main submodules: distpy.util, distpy.transform, distpy.distribution, and distpy.jumping. It is compatible with Python2.7+ and Python3.5+. The submodules are not relevant for the purpose of imports. Any class or function, 'XYZ', described below can be imported using `from distpy import XYZ`. The lists below enumerate the major classes and groups of classes. Below that is a detailed description of each submodule.

Utility interfaces:
    class Savable: Parent class for objects which can be saved to an hdf5 file group
    class Loadable: Parent class for objects which can be loaded from an existing hdf5 file group

hdf5 Utility objects and functions:
    class HDF5Link: Allows for links within hdf5 files to avoid redundant saving
    function create_hdf5_dataset: Creates an hdf5 dataset, possibly with links
    function get_hdf5_value: Gets an hdf5 file from an h5py.Group or h5py.Dataset
    function save_dictionary: Saves a dictionary to an hdf5 group as much as possible.
    function load_dictionary: Loads a dictionary from an hdf5 group that was initially saved using the save_dictionary function

General purpose utility objects:
    class Expression: object that allows for evaluation of Python code strings programmatically through function calling syntax

Transformations: Classes involving univariate transformations, which can be used to define novel transformed distributions
    Utility functions: functions that load transforms from hdf5 files and groups, functions that cast to transforms or collections of transforms, and functions that invert transforms
        load_transform_from_hdf5_group
        load_transform_from_hdf5_file
        cast_to_transform
        castable_to_transform
        invert_transform
    Transform classes: The classes that actually implement the transformations
        AffineTransform
        ArcsinTransform
        ArsinhTransform
        BoxCoxTransform
        CompositeTransform
        Exp10Transform
        ExponentialTransform
        ExponentiatedTransform
        Log10Transform
        LoggedTransform
        LogisticTransform
        LogTransform
        NullTransform
        PowerTransform
        ProductTransform
        ReciprocalTransform
        SineTransform
        SumTransform
    Transform container classes: Ordered or unordered collections of Transform objects
        TransformList
        TransformSet

Distributions: Classes involving distributions of variables
    Utility functions: functions that load distributions from hdf5 files and groups
        load_distribution_from_hdf5_group
        load_distribution_from_hdf5_file
    Discrete distribution classes: Distributions defined at discrete points (usually the integers), univariate or multivariate
        BernoulliDistribution
        BinomialDistribution
        CustomDiscreteDistribution
        DiscreteUniformDistribution
        GeometricDistribution
        PoissonDistribution
    Continuous distribution classes: Distributions defined on the real numbers (or a subset), univariate or multivariate
        BetaDistribution
        ChiSquaredDistribution
        DoubleSidedExponentialDistribution
        EllipticalUniformDistribution
        ExponentialDistribution
        GammaDistribution
        GaussianDirectionDistribution
        GaussianDistribution
        GeneralizedParetoDistribution
        GriddedDistribution
        LinearDirectionDistribution
        ParallelepipedDistribution
        SechDistribution
        SechSquaredDistribution
        TruncatedGaussianDistribution
        UniformConditionDistribution
        UniformDirectionDistribution
        UniformDistribution
        UniformTriangulationDistribution
        WeibullDistribution
    Distribution classes that can be discrete or continuous: Distributions whose discrete/continuous nature is determined by inputs
        DeterministicDistribution
        DistributionSum
        InfiniteUniformDistribution
        KroneckerDeltaDistribution
        LinkedDistribution
        SequentialDistribution
        WindowedDistribution
    Distribution container classes: Ordered or unordered collections of Distribution objects
        DistributionList
        DistributionSet
    class DistributionHarmonizer: Creates a joint DistributionSet from a marginal DistributionSet and a conditional DistributionSet

Proposal distributions: Classes involving proposal distributions for random walks, which are a specific class of conditional distributions
    Utility functions: functions that load jumping/proposal distributions from hdf5 files and groups
        load_jumping_distribution_from_hdf5_group
        load_jumping_distribution_from_hdf5_file
    Discrete proposal distribution classes: Proposal distributions defined on integers (or a subset), univariate or multivariate
        AdjacencyJumpingDistribution
        BinomialJumpingDistribution
        GridHopJumpingDistribution
    Continuous proposal distribution classes: Proposal distributions defined on real numbers (or a subset), univariate or multivariate
        GaussianJumpingDistribution
        SourceDependentGaussianJumpingDistribution
        TruncatedGaussianJumpingDistribution
        UniformJumpingDistribution
    Proposal distribution classes that can be discrete or continuous: Proposal distributions whose discrete/continuous nature is determined by inputs
        JumpingDistributionSum
        LocaleIndependentJumpingDistribution
        SourceIndependentJumpingDistribution
    Proposal distribution container classes: Ordered or unordered collections of JumpingDistribution objects
        JumpingDistributionList
        JumpingDistributionSet
    class MetropolisHastingsSampler: Implementation of Metropolis Hastings Markov Chain Monte Carlo sampler



---------------------
distpy.util submodule
---------------------

The distpy.util submodule has some general purpose tools. For example, it contains types for type checking. Specifically, it defines bool_types, int_types, float_types, real_numerical_types, complex_numerical_types, numerical_types, and sequence_types. Any of these can be used to check for the given type of an object x by using (type(x) in XYZ_types).

The distpy.util.submodule also contains the following functions and classes for hdf5 usage:

Class: Savable
Signature: (cannot be directly instantiated)
Description: This is a base class that cannot be directly instantiated. It contains a method, save(file_name), which calls the subclass' implementation of the fill_hdf5_group(group) function (which, on its own, can be used to save the object into an extant h5py.Group) to save the object into a new hdf5 file at file_name.
Function save(file_name): Calls fill_hdf5_group with a group resulting from creating and opening a new hdf5 file at the given file name.

Class: Loadable
Signature: (cannot be directly instantiated)
Description: This is a base class that cannot be directly instantiated. It contains a method, load(file_name), which calls the subclass' implementation of the load_hdf5_group(group) function (which, on its own, can be used to load the object from an extant h5py.Group) to load the object from an extant hdf5 file at file_name.
Method load(file_name): Calls load_hdf5_group with a group resulting from opening and hdf5 file at the given file name.

Class: HDF5Link
Signature: HDF5Link(link, slices=None)
Description: A soft or hard link in an hdf5 file. link can be either a h5py.Dataset object, an h5py.Group object, or a string absolute path within a file. slices allows for datasets to be sliced in links.

Function: create_hdf5_dataset
Signature: create_hdf5_dataset(group, name, data=None, link=None)
Description: group is the h5py.Group object in which to place the dataset. name is the string identifier of the dataset in group. At least one of data or link is not None. If data is given, it should be the np.ndarray of data to put into the dataset. link should be an HDF5Link or a h5py.Group object if given. If link and data are both used, data is not used.

Function: get_hdf5_value
Signature: get_hdf5_value(obj)
Description: obj should be an object created with the create_hdf5_dataset function described above.

Function: save_dictionary
Signature: save_dictionary(dictionary, group)
Description: Saves the given dictionary to the given h5py.Group.

Function: load_dictionary
Signature: load_dictionary(group, **classes_to_load)
Description: Loads a dictionary from the given group. classes_to_load should be a dictionary with string paths as keys and class objects to load.


Other utility:

Class: Expression
Signature: Expression(string, num_arguments=None, import_strings=[], kwargs={}, should_throw_error=False)
Description: The Expression class is a way to evaluate strings with arguments. string is the Python code to evaluate. num_arguments is the integer number of arguments to accept. import_strings allows for imports to be done before evaluation. kwargs is a dictionary to use when formatting string.

There are also univariate_histogram, bivariate_histogram, and triangle_plot functions to plot distributions/samples.


Plotting samples:

Function: univariate_histogram
Signature: univariate_histogram(sample, reference_value=None, bins=None, matplotlib_function='fill_between', show_intervals=False, xlabel='', ylabel='', title='', fontsize=28, ax=None, show=False, norm_by_max=True, **kwargs)
Description: Plots a univariate histogram given the 1D array sample. A dashed vertical line is plotted at reference_value, if it is given. bins determines the bins used by the histogram maker (it can be an integer number of bins or a sequence of bin edges. The matplotlib_function used can be 'plot', 'fill_between', or 'bar'. If show_intervals is True, a 95% confidence interval is plotted. xlabel, ylabel, title, and fontsize determine the text drawn on the plot. ax can be an existing matplotlib.pyplot.Axes object on which to plot or can be None if a new one should be created. show allows the user to determine if matplotlib.pyplot.show() function should be called before the function returns. norm_by_max allows control over whether the y values of the histograms are the number of occurrences or the number of occurrences divided by the number of occurrences in the bin with the most points. kwargs should be keyword arguments that are passed to the given matplotlib_function. Returns the matplotlib.pyplot.Axes object on which this histogram is plotted if show is False.

Function: bivariate_histogram
Signature: bivariate_histogram(xsample, ysample, reference_value_mean=None, reference_value_covariance=None, bins=None, matplotlib_function='imshow', xlabel='', ylabel='', title='', fontsize=28, ax=None, show=False, contour_confidence_levels=0.95, reference_color='r', reference_alpha=1, **kwargs)
Description: Plots a bivariate histogram using the given xsample and ysample 1D arrays. A dashed horizontal/vertical cross is plotted at reference_value_mean if it is given. If reference_value_mean and reference_value_covariance are both defined, then an ellipse is plotted corresponding to a 2D Gaussian distribution with that mean and covariance at the probability level given by the maximum of the given contour_confidence_levels. bins can be either an integer number of bins or a tuple of the form (xbins, ybins) where xbins and ybins can be either integer numbers of bins in each dimension or sequences of bin edges in each dimension. The matplotlib_function used can be either 'imshow', which shows the histogram directly or 'contour' or 'contourf', which plot confidence level contours given by contour_confidence_levels (which should be either a single probability level or an increasing sequence of probability levels). reference_color and reference_alpha determine the appearance of the reference value ellipse. xlabel, ylabel, title, and fontsize determine the text drawn on the plot. ax can be an existing matplotlib.pyplot.Axes object on which to plot the histogram or None if a new one should be created. show allows the user to control whether matplotlib.pyplot.show is called before the function returns. kwargs are keyword arguments that are passed on to the given matplotlib_function. Returns the matplotlib.pyplot.Axes object on which this histogram is plotted if show is False.

Function: triangle_plot
Signature: triangle_plot(samples, labels, figsize=(8, 8), fig=None, show=False, kwargs_1D={}, kwargs_2D={}, fontsize=28, nbins=100, plot_type='contour', reference_value_mean=None, reference_value_covariance=None, contour_confidence_levels=0.95, tick_label_format_string='{x:.3g}')
Description: Plots a triangle plot, which is a matrix of plots with univariate histograms on the diagonal and bivariate histograms below the diagonal, using the given samples, which should be given as a 2D array of shape (num_params, num_draws). labels is a sequence of strings of length num_params which will determine the labels on the plots. figsize should be a tuple of form (length_in_inches, width_in_inches) and is only used if fig is None. fig can be an existing matplotlib.pyplot.Figure object on which to plot the histogram or None if a new one should be created. show allows the user to determine whether matplotlib.pyplot.plot function is called before this function returns. nbins should be an integer number of bins to use in each dimension. plot_type should be one of 'contour', 'contourf', or 'histogram'. If either of the first two is used, than the matplotlib_function passed to univariate_histogram is 'plot', whereas if the last one is used, than the matplotlib_function passed to univariate_histogram is 'bar'. If 'contour' or 'contourf' is chosen, it is passed as the matplotlib_function to bivariate_histogram. If 'histogram' is chosen, then the matplotlib_function passed to bivariate_histogram is 'imshow'. If reference_value_mean is given, dashed crosses are plotted at the values. If reference_value_mean and reference_value_covariance are both given, crosses are drawn and confidence ellipses are plotted (corresponding to maximum probability level of contour_confidence_levels) for each bivariate histogram. The tick_label_format_string is a string which can be formatted using tick_label_format_string.format(x=value), where value is the location of the tick, returning the string that should be placed at the tick. kwargs_1D are extra keyword arguments passed to the univariate_histogram function and kwargs_2D are extra keyword arguments passed to the bivariate_histogram function. Returns the matplotlib.pyplot.Figure object on which this histogram is plotted if show is False.



--------------------------
distpy.transform submodule
--------------------------

The distpy.transform submodule defines the below monotonic variable transformations. Each transformation is given by its initialization signature, equation of the form y = f(x), where x (y) is the input (output) of the transformation. All transforms are subclasses of the Transform class, which is the first class described below.

Class: Transform
Description: Base class for all Transform objects. Requires subclasses to implement the following functions:
Method apply(x): applies the transformation to the value x
Method apply_inverse(x): applies the inverse of the transformation to the value x
Method derivative(x): applies the derivative of the transformation to the value x
Method log_derivative(x): applies the natural log of the derivative of the transformation to the value x
Method second_derivative(x): applies the second derivative of the transformation to the value x
Method derivative_of_log_derivative(x): applies the derivative of the log of the derivative of the transformation to the value x
Method third_derivative(x): applies the third derivative of the transformation to the value x
Method second_derivative_of_log_derivative(x): applies the second derivative of the log of the derivative of the transformation to the value x
Method load_from_hdf5_group(group): staticmethod implemented by all subclasses that loads the Transform from a group of an hdf5 file.
Method fill_hdf5_group(group): fills the given group with information about this transformation so that load_from_hdf5_group can be called later

Class: AffineTransform
Signature: AffineTransform(scale_factor, translation)
Equation: y = (scale_factor * x) + translation
Domain: [-inf, +inf]
Range: [-inf, +inf]

Class: ArcsinTransform
Signature: ArcsinTransform()
Equation: y = arcsin(x)
Domain: [-1, +1]
Range: [-pi/2, +pi/2]

Class: ArsinhTransform
Signature: ArsinhTransform(shape)
Equation: y = (x if (shape == 0) else ((sinh(shape * x) / shape) if (shape > 0) else (arcsinh(shape * x) / shape)))
Domain: [-inf, +inf]
Range: [-inf, +inf]

Class: BoxCoxTransform
Signature: BoxCoxTransform(power, offset=0)
Equation: y = (ln(x + offset) if (power == 0) else ((((x + offset) ** power) - 1) / power))
Domain: [offset, +inf]
Range: [-1/power, +inf]

Class: CompositeTransform
Signature: CompositeTransform(inner_transform, outer_transform)
Equation: y = outer_transform(inner_transform(x))
Domain: Domain of inner transform
Range: Range of outer transform

Class: Exp10Transform
Signature: Exp10Transform()
Equation: y = 10 ** x
Domain: [-inf, +inf]
Range: [0, +inf]

Class: ExponentialTransform
Signature: ExponentialTransform()
Equation: y = e ** x
Domain: [-inf, +inf]
Range: [0, +inf]

Class: ExponentiatedTransform
Signature: ExponentiatedTransform(transform)
Equation: y = e ** transform(x)
Domain: Domain of transform
Range: [0, +inf]

Class: LogTransform
Signature: LogTransform()
Equation: y = ln(x)
Domain: [0, +inf]
Range: [-inf, +inf]

Class: Log10Transform
Signature: Log10Transform()
Equation: y = ln(x) / ln(10)
Domain: [0, +inf]
Range: [-inf, +inf]

Class: LoggedTransform
Signature: LoggedTransform(transform)
Equation: y = ln(transform(x))
Domain: Domain of transform
Range: [-inf, +inf]

Class: LogisticTransform
Signature: LogisticTransform()
Equation: y = ln(x / (1 - x))
Domain: [0, +1]
Range: [-inf, +inf]

Class: NullTransform
Signature: NullTransform()
Equation: y = x
Domain: [-inf, +inf]
Range: [-inf, +inf]

Class: PowerTransform
Signature: PowerTransform(power)
Equation: y = x ** power
Domain: [0, +inf]
Range: [0, +inf]

Class: ProductTransform
Signature: ProductTransform(first_transform, second_transform)
Equation: y = first_transform(x) * second_transform(x)
Domain: Union of domains of first_transform and second_transform
Range: Product of ranges of first_transform and second_transform

Class: ReciprocalTransform
Signature: ReciprocalTransform()
Equation: y = 1 / x
Domain: [-inf, +inf] (monotonic only if used in either [-inf, 0] or [0, +inf])
Range: [-inf, +inf]

Class: SineTransform
Signature: SineTransform()
Equation: y = sin(x)
Domain: [-pi/2, +pi/2]
Range: [-1, +1]

Class: SumTransform
Signature: SumTransform(first_transform, second_transform)
Equation: y = first_transform(x) + second_transform(x)
Domain: Union of domains of first_transform and second_transform
Range: Sum of ranges of first_transform and second_transform


The distpy.transform submodule also defines two different collections designed specifically to store Transform objects.

Class: TransformList
Signature: TransformList(*transforms)
Initialization description: The TransformList class is initialized very simply by providing an arbitrary amount of positional arguments, all of which are Transform objects or things that can be cast to Transform objects (see description of cast_transform below).
Class description: TransformList is a list-like object containing transforms, which can be called through transform_list(inputs), which is a convenient way to run each transform on each input (where inputs is array-like), outputting an array-like output.
Method append(transform): Appends a Transform to the list, adding it to the end
Method extend(transform_list): Extends this TransformList with another TransformList (or something that can be cast into one) by adding it onto the end
Method apply(point, axis=-1): Applies the transformations in the TransformList to a point along the given axis
Method apply_inverse(point, axis=-1): Applies the inverses of the transformations in the TransformList to a point along the given axis
Method cast(key, num_transforms=None): staticmethod that casts the key into a TransformList
Method castable(key, num_transforms=None, return_transform_list_if_true=False): staticmethod that checks if key can be cast into a TransformList
Method derivative(point, axis=-1): Applies the derivatives of the transformations in the TransformList to a point along the given axis
Method log_derivative(point, axis=-1): Applies the logs of the derivatives of the transformations in the TransformList to a point along the 
given axis
Method second_derivative(point, axis=-1): Applies the second derivatives of the transformations in the TransformList to a point along the given axis
Method derivative_of_log_derivative(point, axis=-1): Applies the derivatives of the logs of the derivatives of the transformations in the TransformList to a point along the given axis
Method third_derivative(point, axis=-1): Applies the third derivatives of the transformations in the TransformList to a point along the given axis
Method second_derivative_of_log_derivative(point, axis=-1): Applies the second derivatives of the logs of the derivatives of the transformations in the TransformList to a point along the given axis
Method transform_gradient(untransformed_gradient, untransformed_point, axis=-1): Changes the gradient evaluated at the given point so that it is expressed in transformed space
Method detransform_gradient(transformed_gradient, untransformed_point, axis=-1): Changes the transformed gradient evaluated at the transformed version of untransformed_point into untransformed space
Method transform_hessian(untransformed_hessian, transformed_gradient, untransformed_point, first_axis=-2): Changes the hessian evaluated at the given point so that it is expressed in transformed space
Method detransform_hessian(transformed_hessian, transformed_gradient, untransformed_point, axis=-1): Changes the transformed hessian evaluated at the transformed version of untransformed_point into untransformed space, using the transformed gradient evaluated at the same point
Method transform_covariance(untransformed_covariance, untransformed_point, axis=(-2, -1)): Changes the covariance into untransformed space. This assumes that the covariance is small enough for derivatives to completely determine its changes
Property is_null: Checks to see if all transformations in this TransformList are null transformations

Class: TransformSet
Signature: TransformSet(transforms, parameters=None)
Initialization description: The TransformSet class can be initialized in one of two ways. First, transforms can be a string-keyed dictionary with Transform objects or things which can be cast to Transform objects (see description of cast_transform below) as values, in which case the parameters initialization argument is unused. Second, transforms can be a sequence of Transform objects or things which can be cast to Transform objects (see description of cast_transform below), in which case the parameters initialization argument must be a sequence of string names corresponding to those transforms. In this second case, initializing a TransformSet through TransformSet(transforms, parameters) is equivalent to initializing it through TransformSet(dict(zip(parameters, transforms))).
Class description: TransformSet is a dictionary-like object containing transforms, which can be called through transform_set(inputs), which is a convenient way to run each transform on each input (where inputs is dictionary-like), outputting a dictionary output.


The distpy.transform submodule contains a few utility functions:

Function: load_transform_from_hdf5_group
Signature: load_transform_from_hdf5_group(group)
Description: Loads a Transform (of unknown type) from the given h5py.Group.

Function: load_transform_from_hdf5_file
Signature: load_transform_from_hdf5_file(file_name)
Description: Loads a Transform (of unknown type) from the extant hdf5 file at the given file name.

Function: invert_transform
Signature: invert_transform(transform)
Description: Returns a Transform object that inverts transform.

Function: cast_transform
Signature: cast_transform(key)
Description: If key is a Transform object it is returned directly. If key is None, a NullTransform object is returned. If key is a single token string (i.e. having no spaces), it is interpreted as the name of the transform to include according to the following mapping (all tokens are case-independent):
    ArcsinTransform: ['arcsin']
    ExponentialTransform: ['exp']
    Exp10Transform: ['exp10']
    LogTransform: ['log', 'ln']
    Log10Transform: ['log10']
    LogisticTransform: ['logistic']
    NullTransform: ['none', 'null']
    ReciprocalTransform: ['reciprocal']
    SineTransform: ['sine']
If key is a multi-token string (when split with spaces), it is cast according to the following mapping (again, all tokens are case-independent) where bash-like references with '$' are used to indicate numerical values:
    AffineTransform($SCALEFACTOR, 0): ['scale $SCALEFACTOR']
    AffineTransform(1, $TRANSLATION): ['translate $TRANSLATION']
    AffineTransform($SCALEFACTOR, $TRANSLATION): ['affine $SCALEFACTOR $TRANSLATION']
    ArsinhTransform($SHAPE): ['arsinh $SHAPE']
    BoxCoxTransform($POWER, offset=0): ['boxcox $POWER', 'box-cox $POWER']
    BoxCoxTransform($POWER, offset=$OFFSET): ['boxcox $POWER $OFFSET', 'box-cox $POWER $OFFSET']
    PowerTransform($POWER): ['power $POWER']

Function: castable_to_transform
Signature: castable_to_transform(key, return_transform_if_true=False)
Description: Attempts to cast key to a Transform object using the cast_transform function described above. If key cannot be cast to a Transform, then False is returned. If key can be cast to a Transform denoted by transform, then (transform if return_transform_if_true else True) is returned.



-----------------------------
distpy.distribution submodule
-----------------------------

The distpy.distribution module is the main purpose of distpy. It contains a base class called Distribution, which creates an interface with draw, log_value, to_string, and fill_hdf5_group methods and a numparams property all to be implemented in subclasses. It also provides many specific implemented subclasses of the Distribution class, described below.

Class: Distribution
Signature: (Distribution class cannot be directly instantiated)
Description: Base class for all distributions. All Distribution objects can store metadata passed in their initializations and can be saved and loaded to hdf5 file/group objects because they implement the Savable and Loadable classes from the distpy.util module.
Method draw(shape=None, random=np.random): Allows for random values to be drawn from the distribution in the given shape (None shape indicates a single random variate). random keyword argument allows for the passing of a mtrand random state.
Method log_value(point): Computes the log density of points drawn around the given point. -np.inf indicates a 0 probability.
Method: gradient_of_log_value(point): Computes the derivative vector of the log density around the given point. This can only be done if the distribution is continuous and the gradient_computable property is True, which it is for most analytical distributions.
Method: hessian_of_log_value(point): Computes the second derivative matrix of the log density around the given point. This can only be done if the distribution is continuous and the hessian_computable property is True, which it is for most analytical distributions.
Method left_confidence_interval(probability_level): determines the leftmost non-disjoint interval that contains the variable of this distribution at the given probability_level. Can only be called without error if can_give_confidence_intervals property is True, which is only for analytical continuous univariate distributions. Essentially, this method produces the tuple (Finv(0), Finv(probability_level)) where Finv is the inverse of the cumulative distribution function.
Method central_confidence_interval(probability_level): determines the central (around the median) interval that contains the variable of this distribution at the given probability_level. Can only be called without error if can_give_confidence_intervals property is True, which is only for analytical continuous univariate distributions. Essentially, this method produces the tuple (Finv((1-probability_level)/2), Finv((1+probability_level)/2)) where Finv is the inverse of the cumulative distribution function.
Method right_confidence_interval(probability_level): determines the rightmost interval that contains the variable of this distribution at the given probability_level. Can only be called without error if can_give_confidence_intervals property is True, which is only for analytical continuous univariate distributions. Essentially, this method produces the tuple (Finv(1-probability_level), Finv(1) where Finv is the inverse of the cumulative distribution function.
Method to_string(): Returns a string description of this distribution.
Method plot(self, x_values, scale_factor=1, xlabel='', ylabel='', title='', fontsize=24, ax=None, show=False, **kwargs): Plots the distribution at the given x_values. Uses a matplotlib.pyplot.scatter if the distribution is discrete and matplotlib.pyplot.plot if the distribution is continuous.
Method copy(): Returns a deep copy of this distribution.
Method __len__(): Allows for checking the number of parameters of Distribution objects using the len function.
Method __eq__(other): Allows for equality checking of Distribution objects using the '==' symbol.
Method __ne__(other): Allows for inequality checking of Distribution objects using the '!=' symbol. This automatically returns the opposite of the __eq__ method.
Property mean: Number or array storing mean of this distribution.
Property variance: Number or array storing (co)variance of this distribution.
Property numparams: Integer number of parameters described by the distribution.
Property maximum: The maximum value(s) of the parameter(s) of this distribution.
Property minimum: The minimum value(s) of the parameter(s) of this distribution.
Property is_discrete: Boolean describing whether log_value corresponds to a discrete probability mass function or a continuous probability density function.
Property gradient_computable: Describes whether gradient_of_log_value method can be called meaningfully without throwing an error.
Property hessian_computable: Describes whether hessian_of_log_value method can be called meaningfully without throwing an error.
Property can_give_confidence_intervals: Describes whether this distribution can yield confidence intervals. This is True for most continuous univariate distributions.

Class: BernoulliDistribution
Signature: BernoulliDistribution(probability_of_success, metadata=None)
Mass: p(x) = ((1 - probability_of_success) if (x == 0) else probability_of_success)
Domain: Integers 0 and 1
Mean: probability_of_success
Variance: probability_of_success * (1 - probability_of_success)
Description/Notes: Simplest nontrivial discrete distribution.

Class: BetaDistribution
Signature: BetaDistribution(alpha, beta, metadata=None)
Density: p(x) = (x ** (alpha - 1)) * ((1 - x) ** (beta - 1)) * Gamma(alpha + beta) / (Gamma(alpha) * Gamma(beta))
Domain: Real numbers [0, 1]
Mean: alpha / (alpha + beta)
Variance: (alpha * beta) / (((alpha + beta) ** 2) * (alpha + beta + 1))
Description/Notes: alpha and beta must both be positive real numbers.

Class: BinomialDistribution
Signature: BinomialDistribution(probability_of_success, number_of_trials, metadata=None)
Mass: p(x) = (probability_of_success ** x) * ((1 - probability_of_success) ** (number_of_trials - x)) * Gamma(number_of_trials + 1) / (Gamma(x + 1) * Gamma(number_of_trials - x + 1))
Domain: Integers [0, number_of_trials]
Mean: number_of_trials * probability_of_success
Variance: number_of_trials * probability_of_success * (1 - probability_of_success)
Description/Notes: probability_of_success is a real number between 0 and 1 (exclusive) and number of trials is a positive integer. The binomial distribution describes the probability of succeeding x times out of number_of_trials when the probability of success on a given run is probability_of_success.

Class: ChiSquaredDistribution
Signature: ChiSquaredDistribution(degrees_of_freedom, reduced=False, metadata=None)
Density: p(x) = ((e ** (-((norm * x) / 2))) * (((norm * x) / 2) ** ((degrees_of_freedom / 2) - 1))) * (norm / (2 * Gamma(degrees_of_freedom / 2))) where norm = (degrees_of_freedom if reduced else 1)
Domain: Real numbers [0, +inf]
Mean: (1 if reduced else degrees_of_freedom)
Variance: (2 / degrees_of_freedom) if reduced else (2 * degrees_of_freedom)
Description/Notes: degrees_of_freedom is a positive integer. If reduced is True, this is the distribution of (chi_squared / degrees_of_freedom) where chi_squared is the sum of the squares of degrees_of_freedom centered, scaled normal values.

Class: CustomDiscreteDistribution
Signature: CustomDiscreteDistribution(variable_values, probability_mass_function, metadata=None)
Mass: p(x) = probability_mass_function[k_1,k_2,...,k_N] array element at indices corresponding to variable values
Domain: Grid defined by the variable_values, which is a sequence of N 1D numpy.ndarrays.
Mean: Depends on probability_mass_function
Variance: Depends on probability_mass_function
Description/Notes: This class is designed to allow for the efficient usage of arbitrary discrete distributions, which can be univariate or multivariate.

Class: DeterministicDistribution
Signature: DeterministicDistribution(points, is_discrete=False, metadata=None)
Density/Mass: Undefined because only a finite sample is known. As points are drawn, they are taken directly from the points array.
Domain: N/A
Mean: N/A
Variance: N/A
Description/Notes: This class allows for points to be drawn deterministically. If points is 1D, this is assumed to be a set of len(points) single random variates. If it is 2D, it is assumed to be a set of points.shape[0] realizations of points.shape[1]-dimensional variates.

Class: DiscreteUniformDistribution
Signature: DiscreteUniformDistribution(low, high=None, metadata=None)
Note: if high is None, then high is set to low and low is assumed to be zero.
Mass: p(x) = 1 / (high - low + 1)
Domain: Integers [low, high]
Mean: (low + high) / 2
Variance: (((high - low + 1) ** 2) - 1) / 12
Description/Notes: This distribution allows for integers to be drawn uniformly from low to high (inclusive).

Class: DistributionSum
Signature: DistributionSum(distributions, weights, metadata=None)
Density/mass: weighted sum of underlying distributions, renormalized
Domain: Combination of domains of underlying distributions
Mean: weighted sum of means of underlying distributions
Variance: no simple expression, depends on underlying expressions
Description/Notes: This class allows for a distribution which is a weighted sum of many distributions.

Class: DoubleSidedExponentialDistribution
Signature: DoubleSidedExponentialDistribution(mean, variance, metadata=None)
Density: p(x) = (k / 2) * (e ** (-(k * |x - mean|))) where k = (1 / sqrt(variance / 2))
Domain: Real numbers[-inf, +inf]
Mean: mean
Variance: variance
Description/Notes: This is a peaky univariate distribution described by its mean and variance. mean can be any real number but variance must be positive.

Class: EllipticalUniformDistribution
Signature: EllipticalUniformDistribution(mean, covariance, metadata=None)
Density: p(x) = (Gamma(1 + (d / 2)) * (((d + 2) * pi) ** (d / 2))) / sqrt(det(cov)) where d = len(mean)
Domain: Real vectors x satisfying np.dot(np.dot((x - mean).T, inv(cov)), (x - mean)) < d + 2 where d = len(mean)
Mean: mean
Variance: covariance
Description/Notes: This is a uniform distribution on the ellipse centered on mean with shape given by covariance.

Class: ExponentialDistribution
Signature: ExponentialDistribution(rate, shift=0, metadata=None)
Density: p(x) = rate * (e ** (-(rate * (x - shift))))
Domain: Real numbers [shift, +inf]
Mean: shift + (1 / rate)
Variance: (1 / (rate ** 2))
Description/Notes: This is the distribution of the time to wait for a uniform probability event.

Class: GammaDistribution
Signature: GammaDistribution(shape, scale=1, power=1, metadata=None)
Density: p(x) = (1 / Gamma(shape / power)) * (power / scale) * ((x / scale) ** (shape - 1)) * (e ** (-((x / scale) ** power)))
Domain: Real numbers [0, +inf]
Mean: scale * (Gamma((shape + 1) / power) / Gamma(shape / power))
Variance: (scale ** 2) * ((Gamma((shape + 2) / power) / Gamma(shape / power)) - ((Gamma((shape + 1) / power) / Gamma(shape / power)) ** 2))
Description/Notes: A generalizable form of the Gamma function that allows for any power in the exponent.

Class: GaussianDistribution
Signature: GaussianDistribution(mean, covariance, metadata=None)
Density = p(x) = (e ** (-(np.dot(np.dot(x - mean, inv(covariance)), x - mean) / 2))) / sqrt(det(2 * pi * covariance))
Domain: All vectors x of same dimension as mean
Mean: mean
Variance: covariance
Description/Notes: The most useful general continous distribution. Supports univariate or multivariate use.

Class: GeneralizedParetoDistribution
Signature: GeneralizedParetoDistribution(shape, location=0, scale=1, metadata=None)
Density: p(x) = ((shape - 1) / scale) * ((1 + ((x - location) / scale)) ** (-shape))) where shape > 1
Domain: Real numbers [location, +inf]
Mean: location + (scale / (shape - 2)) where shape > 2
Variance: ((scale ** 2) * (shape - 1)) / ((shape - 3) * ((shape - 2) ** 2)) where shape > 3
Description/Notes: Power law distribution, useful in many sociological fields. Shape must be greater than 1 and scale must be positive.

Class: GeometricDistribution
Signature: GeometricDistribution(common_ratio, minimum=0, maximum=None, metadata=None)
Mass: p(x) = ((1 - common_ratio) / (1 - (0 if (maximum is None) else (common_ratio ** span)))) * (common_ratio ** (x - minimum)) where span = maximum - minimum + 1
Domain: Integers [minimum, (+inf if (maximum is None) else maximum)]
Mean: minimum + (common_ratio / (1 - common_ratio)) - ((span * (common_ratio ** span)) / (1 - (common_ratio ** span))) where span = maximum - minimum + 1
Variance: No simple expression
Description/Notes: If minimum is 0 and maximum is None, this represents the probability distribution of the number of successes in a row if the probability of success if (1 - common_ratio).

Class: GriddedDistribution
Signature: GriddedDistribution(variables, pdf=None, metadata=None)
Density: Depends on pdf, uniform within grid squares
Domain: Depends on variables
Mean: Depends on pdf
Variance: Depends on pdf
Description/Notes: This is essentially a custom continuous distribution.

Class: InfiniteUniformDistribution
Signature: InfiniteUniformDistribution(ndim=1, minima=None, maxima=None, metadata=None)
Density: (not defined but returns 0 when log_value is called inside domain)
Domain: between minimum and maximum for each variable
Mean: N/A
Variance: N/A
Description/Notes: The infinite uniform distribution is included even though it is an improper distribution so that it can be used as a prior.

Class: KroneckerDeltaDistribution
Signature: KroneckerDeltaDistribution(value, is_discrete=True, metadata=None)
Density: Infinite at value (cannot be evaluated)
Domain: value only
Mean: value
Variance: 0
Description/Notes: The KroneckerDeltaDistribution always returns the same value. Its log_value cannot be defined.

Class: LinkedDistribution
Signature: LinkedDistribution(shared_distribution, numparams, metadata=None)
Density: f(x) = (e ** shared_distribution.log_value(x[0])) * dirac_delta(x[1] - x[0]) * ... * dirac_delta(x[-1] - x[0])
Domain: domain of shared_distribution when all elements of vector x are equal
Mean: mean_of_shared_distribution * np.ones((numparams,))
Variance: variance_of_shared_distribution * np.ones((numparams,) * 2)
Description/Notes: This distribution was created to have a distribution where all parameters are equal and share a common distribution.

Class: ParallelepipedDistribution
Signature: ParallelepipedDistribution(center, face_directions, distances, metadata=None)
Density: Uniform and equal to 1/V when inside the parallelepiped of volume V
Domain: Interior of parallelepiped
Mean: center
Variance: no simple expression
Description/Notes: This distribution allows for uniform sampling of a parallelepiped in arbitrary dimensions.

Class: PoissonDistribution
Signature: PoissonDistribution(scale, metadata=None)
Mass: p(x) = (scale ** x) * (e ** (-scale)) / Gamma(x + 1)
Domain: Integers [0, +inf]
Mean: scale
Variance: scale
Description/Notes: The Poisson distribution is the distribution of event occurrences in a given amount of time when the probability of an event occurring is constant in time and independent of other events.

Class: SechDistribution
Signature: SechDistribution(mean, variance, metadata=None)
Density: p(x) = sech((pi / 2) * ((x - mean) / sigma)) / (2 * sigma) where sigma = sqrt(variance)
Domain: Real numbers [-inf, +inf]
Mean: mean
Variance: variance
Description/Notes: The sech distribution is a univariate bell curve distribution that is slightly peakier than the Gaussian distribution.

Class: SechSquaredDistribution
Signature: SechSquaredDistribution(mean, variance, metadata=None)
Density: p(x) = (sech((x - mean) / scale) ** 2) / (2 * scale) where scale = sqrt(3 * variance) / pi
Domain: real numbers [-inf, +inf]
Mean: mean
Variance: variance
Description/Notes: This is another bell curve slightly peakier than the sech distribution and even peakier than the GaussianDistribution.

Class: SequentialDistribution
Signature: SequentialDistribution(shared_distribution, numparams=2, metadata=None)
Density: p(x) = Gamma(numparams + 1) * p_shared(x[0]) * ... * p_shared(x[-1])
Domain: All elements of vector x must be in domain of shared_distribution and the vector must be in ascending order
Mean: No simple expression
Variance: No simple expression
Description/Notes: The SequentialDistribution class calls a shared_distribution many times and sorts the outputs.

Class: TruncatedGaussianDistribution
Signature: TruncatedGaussianDistribution(mean, variance, low=None, high=None, metadata=None)
Density: p(x) proportional to GaussianDistribution with given mean and variance
Domain: real numbers [(-inf if (low is None) else low), (+inf if (high is None) else high)]
Mean: no simple expression
Variance: no simple expression
Description/Notes: The TruncatedGaussianDistribution has the benefits of both the GaussianDistribution (localization) and the UniformDistribution (hard cutoffs at boundaries)

Class: UniformConditionDistribution
Signature: UniformConditionDistribution(expression, is_discrete=False, metadata=None)
Density: p(x) proportional to 1 when inside domain and 0 when outside domain
Domain: x such that expression(x) is True
Mean: Depends on expression
Variance: Depends on expression
Description/Notes: This distribution allows for evaluation of the distribution but not drawing from it. Good for checking if condition is True using distribution formalism of distpy.

Class: UniformDistribution
Signature: UniformDistribution(low=0, high=1, metadata=None)
Density: p(x) = 1 / (high - low)
Domain: real numbers [low, high]
Mean: (low + high) / 2
Variance: ((high - low) ** 2) / 12
Description/Notes: Simple continuous uniform distribution between low and high.

Class: UniformTriangulationDistribution
Signature: UniformTriangulationDistribution(triangulation=None, points=None, metadata=None)
Density: p(x) proportional to 1 inside the domain and 0 outside the domain
Domain: Uniform inside the convex hull of the points/triangulation
Mean: Depends on points/triangulation
Variance: Depends on points/triangulation
Description/Notes: This distribution allows for uniform sampling within the convex hull of any set of points.

Class: WeibullDistribution
Signature: WeibullDistribution(shape=1, scale=1, metadata=None)
Density: (shape / scale) * ((x / scale) ** (shape - 1)) * (e ** (-((x / scale) ** shape)))
Domain: real numbers [0, +inf]
Mean: scale * Gamma(1 + (1 / shape))
Variance: (scale ** 2) * (Gamma(1 + (2 / shape)) - (Gamma(1 + (1 / shape)) ** 2))
Description/Notes: The WeibullDistribution can have many different shapes depending on the shape parameter.

Class: WindowedDistribution
Signature: WindowedDistribution(background_distribution, foreground_distribution, metadata=None)
Density: p(x) proportional to p_background(x)
Domain: Union of domains of background_distribution and foreground_distribution
Mean: No simple expression, depends on source distributions
Variance: No simple expression, depends on source distributions
Description/Notes: To draw from this distribution, points are drawn from background distribution and are checked if they are in the domain of foreground_distribution.


Class: DirectionDistribution
Signature: (cannot be directly instantiated)
Description: DirectionDistribution is a parent class of a few different Distribution subclasses that draw points from the 2D spherical surface bounding the sphere in 3D. They take in pointing_center arguments, which should be tuples of the form (latitude, longitude), where both are given in degrees.

Class: GaussianDirectionDistribution
Signature: GaussianDirectionDistribution(pointing_center=(90,0), sigma=1, degrees=True, metadata=None)
Density: p(d) proportional to e ** (-(((gamma / sigma) ** 2) / 2)) where cos(gamma) = dot(pointing_center, d)
Domain: Full sphere
Mean: pointing_center
Variance: no simple expression
Description/Notes: Gaussian distribution on the sphere. degrees argument only determines if sigma is given in degrees. pointing_center is always given in degrees.

Class: LinearDirectionDistribution
Signature: LinearDirectionDistribution(central_pointing, phase_delayed_pointing, angle_distribution, metadata=None)
Density: p(d) proportional to angle distribution density as long as d is on line (great circle) connecting central_pointing and phase_delayed_pointing.
Domain: Great circle connecting central_pointing and phase_delayed_pointing
Mean: no simple expression, depends on angle_distribution
Variance: no simple expression, depends on angle_distribution
Description/Notes: Allows for sampling with arbitrary distribution along a great circle instead of the whole spherical surface.

Class: UniformDirectionDistribution
Signature: UniformDirectionDistribution(low_theta=0, high_theta=pi, low_phi=0, high_phi=2*pi, pointing_center=(90,0), psi_center=0, metadata=None)
Density: p(d) proportional to 1 when inside the domain and 0 when outside the domain
Domain: After sphere is rotated so that pointing_center is at (90, 0) and sphere is then rotated through angle psi, the domain is defined in spherical coordinates given the arguments.
Mean: no simple expression, depends on bounds
Variance: no simple expression, depends on bounds
Description/Notes: A simple uniform distribution in polar coordinates on a spherical surface. The default samples uniformly from the entire sphere


The distpy.distribution submodule also contains two different containers for Distribution objects.

Class: DistributionList
Signature: DistributionList(distribution_tuples=[])
Description: The DistributionList is a subclass of the Distribution parent class that allows for many (independent) Distribution objects to be combined into one, possibly in transformed space. As a result, it implements all of the Distribution methods described above. The initialization argument distribution_tuples should be a list of tuples, tup, that could be individually passed to the add_distribution method (see below) through distribution_list.add_distribution(*tup). The DistributionList class can also be initialized with no arguments, in which case the user can add distributions one at a time using the add_distribution method below. The parameters of the distributions remain unnamed because they can be referred to by their index.
Method add_distribution(distribution, transforms=None): Adds a distribution to the list, defined in the given transformed space. transforms should be something that can be cast to a TransformList of length distribution.numparams using the TransformList.cast function described above in the distpy.transform submodule. If no transforms are given, then the distribution takes inputs and puts out outputs points that are in the same space in which the corresponding distribution is defined.
Method __getitem__(which): Allows for the user to access another DistributionList that only includes the distributions specified through square bracket indexing. which can be an integer distribution index, a sequence of integer distribution indices, or a slice of distribution indices.
Method __delitem__(which): Using the same indexing convention as the __getitem__ method, this method allows for deletion of an arbitrary number of the included distributions. It is also a magic method that allows for the del keyword to be used alongside square bracket indexing.
Method continuous_sublist(): Returns a (shallow) copy of this DistributionList containing only the continous distributions.
Method discrete_sublist(): Returns a (shallow) copy of this DistributionList containing only the discrete distributions.
Method copy(): Returns a deep copy of this DistributionList.
Method __add__(other): Magic method allowing for two DistributionList objects to be combined using the '+' symbol.
Method __eq__(other): Allows for equality checking of two DistributionList objects using the '==' symbol.
Method modify_transforms(new_transform_list): Changes the transforms that define the relationship between the space in which points are returned (the untransformed space) and the space in which Distributions are defined (the transformed space). The argument should be a TransformList of length numparams (or something that can be cast to one using the TransformList.cast(key, numparams) method from the distpy.transform submodule).
Method transformed_version(): Returns a version of this DistributionList where both the untransformed and transformed space of the new DistributionList are the transformed space of this DistributionList.
Property transform_list: TransformList object defining the space in which parameters are returned from the underlying distributions.
Property empty: True if and only if the DistributionList has no Distribution objects in it.

Class: DistributionSet
Signature: DistributionSet(distribution_tuples=[])
Description: The DistributionSet class allows for an unordered set of Distribution objects which can be defined in transformed space and can be drawn from and evaluated simultaneously. The initialization argument distribution_tuples should be a list of tuples, tup, that could be individually passed to the add_distribution method (see below) through distribution_set.add_distribution(*tup). The DistributionSet class can also be initialized with no arguments, in which case the user can add distributions one at a time using the add_distribution method below. The parameters are referred to through string names because no indexing is implied in inputs and outputs (both are dictionaries).
Method add_distribution(distribution, params, transforms=None): Adds a distribution to the set, defined in the given transformed space. params should be a sequence of strings of length distribution.numparams, which is used to define the names of the parameters of this DistributionSet. When accepting inputs or putting out outputs, this DistributionSet uses dictionaries whose keys are these strings. transforms should be something that can be cast to a TransformList of length distribution.numparams using the TransformList.cast function described above in the distpy.transform submodule. If no transforms are given, then the distribution takes inputs and puts out outputs points that are in the same space in which the corresponding distribution is defined.
Method find_distribution(parameter): When given a string name in the parameter argument, this method returns a tuple of the form (distribution, index, transform) where distribution is the Distribution object describing this parameter, index is the integer number (starting at 0) of the given parameter in the distribution, and transform is the Transform object defining the space in which Distribution inputs or outputs the parameter.
Method __getitem__(parameter): Magic method alias of find_distribution above that allows for square bracket indexing when searching.
Method delete_distribution(parameter): Deletes the distribution describing the given parameter (this also deletes the distribution of other parameters which share a Distribution object with this one).
Method __delitem__(parameter): Magic method alias of delete_distribution above that allows for square bracket indexing and the use of the del keyword when deleting distributions.
Method __add__(other): Magic method allowing DistributionSet objects to be combined using the '+' symbol.
Method __len__(): Allows user to check the number of parameters in the DistributionSet without explicitly referencing the numparams property.
Method __eq__(other): Allows for equality checking with DistributionSet objects using the '==' symbol.
Method __ne__(other): Allows for inequality checking with DistributionSet objects using the '!=' symbol. Always returns the opposite of __eq__(other)
Method fill_hdf5_group(group): method that saves the DistributionSet to an h5py.Group (such as an open hdf5 file or some directory inside it)
Method save(file_name): method that saves the DistributionSet to a new hdf5 file located at file_name.
Method load_from_hdf5_group(group): Static method that loads and returns a DistributionSet object from a h5py.Group (such as an open hdf5 file or some directory inside it).
Method load(file_name): Static method that loads and returns a DistributionSet object from an hdf5 file located at file_name.
Method draw(shape=None, random=numpy.random): Draws a random sample from the DistributionSet. This function returns a dictionary whose keys are parameter names and whose values are random values of given shape (setting shape to None returns single values as keys). The random keyword argument allows the user to supply a RandomState (the default is to use simply numpy.random).
Method log_value(point): Evaluates the log density of the point (in untransformed space). point should be a dictionary with parameter names as keys and variable values as values.
Method gradient_of_log_value(point): Using the same input convention as log_value (which is the same as the output convention of draw(shape=None)), this function computes the gradient vector of the log density of the points (in untransformed space), with an order determined by the order of parameter names in the params property.
Method hessian_of_log_value(point): Using the same input convention as log_value (which is the same as the output convention of draw(shape=None)), this function computes the hessian matrix of the log density of the points (in untransformed space), with an order determined by the order of parameter names in the params property.
Method continuous_subset(): Returns a version of this DistributionSet where only the continuous parameters/distributions are included.
Method discrete_subset(): Returns a version of this DistributionSet where only the discrete parameters/distributions are included.
Method copy(): Returns a deep copy of this DistributionSet.
Method distribution_list(parameters): Returns a DistributionList including the distributions in the order corresponding to the given parameters.
Method modify_parameters(function): Modifies the names of the parameters of this distribution. The argument should be a Callable which, when called with old parameter names as inputs, returns new parameter names as outputs.
Method modify_transforms(**new_transforms): Changes the transforms that define the relationship between the space in which points are returned (the untransformed space) and the space in which Distributions are defined (the transformed space). The keyword arguments given to this function should include all parameters as keys and the new Transform objects (or things that can be cast to Transform objects using the cast_to_transform(key) method from the distpy.transform submodule) as values.
Method parameter_strings(parameter): Returns informative strings about the given parameter's place in this Distribution and the Transform which defines the space its distribution returns it in (parameter_string, transform_string).
Method transformed_version(): Returns a version of this DistributionSet where both the untransformed space and the transformed space of the new DistributionSet are the transformed space of this DistributionSet.
Method summary_string(): Returns a string which concisely summarizes the place and Transform of each parameter in the Distribution.
Method plot_univariate_histogram(ndraw, parameter, in_transformed_space=True, reference_value=None, bins=None, matplotlib_function='fill_between', show_intervals=False, norm_by_max=True, xlabel='', ylabel='', title='', fontsize=28, ax=None, show=False, contour_confidence_levels=0.95, **kwargs): Plots a univariate histogram of the given parameter using the univariate_histogram function of the distpy.util submodule. See the description of that function for documentation on the keyword_arguments. ndraw should be an integer number of samples to draw to make the histogram.
Method plot_bivariate_histogram(ndraw, parameter1, parameter2, in_transformed_space=True, reference_value_mean=None, reference_value_covariance=None, bins=None, matplotlib_function='imshow', xlabel='', ylabel='', title='', fontsize=28, ax=None, show=False, contour_confidence_levels=0.95, **kwargs): Plots a bivariate histogram of the given parameters using the bivariate_histogram function of the distpy.util submodule. See the description of that function for documentation on the keyword arguments. ndraw should be an integer number of samples to draw to make the histogram.
Method triangle_plot(ndraw, parameters=None, in_transformed_space=True, figsize=(8, 8), fig=None, show=False, kwargs_1D={}, kwargs_2D={}, fontsize=28, nbins=100, plot_type='contour', reference_value_mean=None, reference_value_covariance=None, contour_confidence_levels=0.95, parameter_renamer=(lambda x: x), tick_label_format_string='{x:.3g}'): Makes a triangle plot of the given parameters (None means all parameters are plotted) using the triangle_plot function from the distpy.util submodule. See the description of that function for documentation of the keyword arguments. ndraw should be an integer number of samples to draw to make the histograms.
Property params: Sequence of strings representing the parameters described by this distribution in an internal order. This order is used to define the order of the gradient and hessian values of this DistributionSet
Property numparams: Integer number of parameters described by the DistributionSet.
Property continuous_params: Returns a list of strings representing the continuous parameters described by this distribution.
Property discrete_params: Returns a list of strings representing the discrete parameters described by this distribution.
Property gradient_computable: Boolean describing whether gradient_of_log_value can be called meaningfully without throwing an error.
Property hessian_computable: Boolean describing whether hessian_of_log_value can be called meaningfully without throwing an error.
Property empty: True if and only if the DistributionSet has no Distribution objects in it.
Property minimum: Dictionary containing the minimum values allowable for each parameter indexed by the parameter's name.
Property maximum: Dictionary containing the maximum values allowable for each parameter indexed by the parameter's name.
Property transform_set: TransformSet object yielding the Transform objects corresponding to eahc parameter name.


The distpy.distribution submodule provides the following methods to load distributions from hdf5 files or groups.

Function: load_distribution_from_hdf5_group
Signature: load_distribution_from_hdf5_group(group, *args)
Description: Loads a Distribution object from the given h5py.Group (such as an open hdf5 file or a subdirectory of it), where it was earlier saved using the fill_hdf5_group(group) method of the Distribution class.

Function: load_distribution_from_hdf5_file
Signature: load_distribution_from_hdf5_file(file_name)
Description: Loads a Distribution object from an existing hdf5 file at the given location.


The distpy.distribution submodule provides the following class for approximating a joint distribution from marginal and conditional distributions.

Class: DistributionHarmonizer
Signature: DistributionHarmonizer(marginal_distribution_set, conditional_solver, marginal_draws, conditional_draws=None, **transforms)
Description: The DistributionHarmonizer class creates a sample from a joint distribution by drawing from a marginal distribution many times and drawing from a conditional distribution for each marginal draw. marginal_distribution_set encodes the marginal distribution. marginal_draws is an integer number of times the marginal distribution should be drawn from. conditional_draws is either None or an integer number of times the conditional distribution should be drawn from. If conditional_draws is None, then conditional_solver should be a Callable that should return a dictionary of conditional parameter values solved for when passed a marginal draw (in dictionary form). If conditional_draws is an integer, conditional_solver should be a Callable that should return a DistributionSet representing the conditional distribution when passed a marginal draw (in dictionary form). The keyword arguments transforms should be Transform objects (or things that can be cast to Transform objects) in which the joint distribution should be defined.
Property joint_distribution_set: DistributionSet object containing an approximation to the joint distribution.



------------------------
distpy.jumping submodule
------------------------

The distpy.jumping module is similar to the distpy.distribution module except the distributions it contains require a source point to determine a destination point, as is necessary for random walks like Markov Chain Monte Carlo (MCMC) samplers. It also contains a MetropolisHastingsSampler class which uses these distributions as proposals. It contains a base class called JumpingDistribution, which creates an interface with draw, log_value, log_value_difference, to_string, and fill_hdf5_group methods and a numparams property all to be implemented in subclasses. It also provides many specific implemented subclasses of the JumpingDistribution class, described below.


Class: JumpingDistribution
Signature: (JumpingDistribution class cannot be directly instantiated)
Description: Base class for all jumping/proposal distributions. All JumpingDistribution objects can be saved and loaded to hdf5 file/group objects because they implement the Savable and Loadable classes from the distpy.util module.
Method draw(source, shape=None, random=np.random): Allows for random values to be drawn from the distribution in the given shape, assuming the given source point (None shape indicates a single random variate). random keyword argument allows for the passing of a mtrand random state.
Method log_value(source, destination): Computes the log density of points drawn around the given destination when jumping from the given source. -np.inf indicates a 0 probability.
Method log_value_difference(source, destination): Computes log_value(source,destination)-log_value(destination,source)
Method plot(self, x_values, scale_factor=1, xlabel='', ylabel='', title='', fontsize=24, ax=None, show=False, **kwargs): Plots the distribution at the given x_values. Uses a matplotlib.pyplot.scatter if the distribution is discrete and matplotlib.pyplot.plot if the distribution is continuous.
Method __len__(): Allows for checking the number of parameters of Distribution objects using the len function.
Method __eq__(other): Allows for equality checking of Distribution objects using the '==' symbol.
Method __ne__(other): Allows for inequality checking of Distribution objects using the '!=' symbol. This automatically returns the opposite of the __eq__ method.
Property numparams: Integer number of parameters described by the distribution.
Property is_discrete: Boolean describing whether log_value corresponds to a discrete probability mass function or a continuous probability density function.

Class: AdjacencyJumpingDistribution
Signature: AdjacencyJumpingDistribution(jumping_probability=0.5, minimum=None, maximum=None)
Mass: p(destination|source) = (1 - jumping_probability) if destination and source are the same, jumping_probability is distributed evenly among the destination points one away from source and between minimum and maximum (inclusive)
Description/Notes: This is a 1D discrete jumping distributions defined on a (possibly infinite) subset of the integers. Using it, only jumps of length one are possible and the minimum and maximum allowable values can be imposed efficiently.

Class: BinomialJumpingDistribution
Signature: BinomialJumpingDistribution(minimum, maximum)
Mass: p(destination|source) is the same as (e ** BinomialDistribution(p_value(source), span).log_value(destination - minimum)) where p_value(source) is 1/(2*span) if source is minimum, 1-(1/(2*span)) if source is maximum, or (source/span) if source is neither minimum nor maximum and span = maximum - minimum
Description/Notes: This is another 1D discrete jumping distribution defined on a finite subset of the integers. Unlike the AdjacencyJumpingDistribution, jumps of length greater than 1 are possible. This jumping distribution is based on the (shifted) BinomialDistribution whose mean is source. The only times where this is not rigorously true is when source is the minimum or maximum, in which case the the BinomialDistribution which has a mean 1/2 in from the extremum is used.

Class: GaussianJumpingDistribution
Signature: GaussianJumpingDistribution(covariance)
Density: p(destination|source) = (e ** (-(dot(dot(destination - source, inv(covariance)), destination - source)) / 2)) / sqrt(det(2 * pi * covariance))
Description/Notes: This is the most heavily used proposal distribution in MCMC analysis. It is a continuous distribution whose density at destination is equal to that of a GaussianDistribution with the same covariance and mean equal to source.

Class: GridHopJumpingDistribution
Signature: GridHopJumpingDistribution(ndim=2, jumping_probability=0.5, minima=None, maxima=None)
Mass: p(destination|source) = (1 - jumping_probability) if destination and source are the same, jumping_probability is distributed evenly among the destination points one away from the source and between all minima and all maxima.
Description/Notes: The GridHopJumpingDistribution class generalizes the AdjacencyJumpingDistribution to multiple dimensions. Only one dimension is jumped in at a time.

Class: JumpingDistributionSum
Signature: JumpingDistributionSum(jumping_distributions, weights)
Density/mass: weighted sum of underlying jumping distributions, renormalized
Description/Notes: This class allows for a jumping distribution which is a weighted sum of many jumping distributions.

Class: LocaleIndependentJumpingDistribution
Signature: LocaleIndependentJumpingDistribution(distribution)
Density/Mass: p(destination|source) = (e ** distribution.log_value(destination - source))
Description/Notes: The LocaleIndependentJumpingDistribution generalizes any Distribution object into a JumpingDistribution object by using distribution to determine how far and in which direction the destination should be from the source. Essentially distribution-source has the same density/mass (both discrete and continuous distributions can be made this way) as the given Distribution. For example, GaussianJumpingDistribution(covariance) could be conceptually (i.e. not practically/efficiently) recreated through LocaleIndependentJumpingDistribution(GaussianDistribution(np.zeros(numparams), covariance))

Class: SourceDependentGaussianJumpingDistribution
Signature: SourceDependentGaussianJumpingDistribution(points, covariances)
Density: p(destination|source) = (e ** GaussianJumpingDistribution(covariance(source)).log_value(destination - source)) where covariance(source) is the element of covariances whose corresponding element of points is closest to source.
Description/Notes: This is a first attempt at the creation of a multi-Gaussian jumping distribution, which would be needed in the case of exploring a distribution that is highly non-Gaussian. For a given source, this jumping distribution is Gaussian, but the covariance used is dependent on which of points is closest to source.

Class: SourceIndependentJumpingDistribution
Signature: SourceIndependentJumpingDistribution(distribution)
Density/Mass: p(destination|source) = (e ** distribution.log_value(destination))
Description/Notes: The SourceIndependentJumpingDistribution is another JumpingDistribution seeded by a normal Distribution class. In this case, the density does not depend on the source at all and the destination is merely distributed according to the given Distribution.

Class: TruncatedGaussianJumpingDistribution
Signature: TruncatedGaussianJumpingDistribution(variance, low=None, high=None)
Density: p(destination|source) is proportional to (e ** (-((destination - source) ** 2)/(2 * variance))) inside the domain, and 0 outside. The domain is between (-np.inf if (low is None) else low) and (+np.inf if (high is None) else high)
Description/Notes: This is a 1D GaussianJumpingDistribution which is not allows to exit a specific (possibly infinite) domain between low and high.

Class: UniformJumpingDistribution
Signature: UniformJumpingDistribution(covariance)
Density: p(destination|source) is the same as (e ** UniformDistribution(source - sqrt(3 * covariance), source + sqrt(3 * covariance)).log_value(destination)) if univariate and is the same as (e ** EllipticalUniformDistribution(source, covariance).log_value(destination)) if multivariate
Description/Notes: This class represents a jumping distribution where the destination is distributed through a UniformDistribution centered at source. It can be either univariate or multivariate.


The distpy.jumping submodule also contains two different containers for JumpingDistribution objects.

Class: JumpingDistributionList
Signature: JumpingDistributionList(jumping_distribution_tuples=[])
Description: The JumpingDistributionList is a subclass of the JumpingDistribution parent class that allows for many (independent) JumpingDistribution objects to be combined into one, possibly in transformed space. As a result, it implements all of the JumpingDistribution methods described above. The initialization argument jumping_distribution_tuples should be a list of tuples, tup, that could be individually passed to the add_distribution method (see below) through jumping_distribution_list.add_distribution(*tup). The JumpingDistributionList class can also be initialized with no arguments, in which case the user can add jumping distributions one at a time using the add_distribution method below. The parameters of the jumping distributions remain unnamed because they can be referred to by their index.
Method add_distribution(jumping_distribution, transforms=None): Adds a jumping distribution to the list, defined in the given transformed space. transforms should be something that can be cast to a TransformList of length jumping_distribution.numparams using the TransformList.cast function described above in the distpy.transform submodule. If no transforms are given, then the jumping distribution takes inputs and puts out outputs points that are in the same space in which the corresponding jumping distribution is defined.
Method __getitem__(which): Allows for the user to access another JumpingDistributionList that only includes the jumping distributions specified through square bracket indexing. which can be an integer jumping distribution index, a sequence of integer jumping distribution indices, or a slice of jumping distribution indices.
Method __delitem__(which): Using the same indexing convention as the __getitem__ method, this method allows for deletion of an arbitrary number of the included jumping distributions. It is also a magic method that allows for the del keyword to be used alongside square bracket indexing.
Method continuous_sublist(): Returns a (shallow) copy of this JumpingDistributionList containing only the continous jumping distributions.
Method discrete_sublist(): Returns a (shallow) copy of this JumpingDistributionList containing only the discrete jumping distributions.
Method __add__(other): Magic method allowing for two JumpingDistributionList objects to be combined using the '+' symbol.
Method __eq__(other): Allows for equality checking of two JumpingDistributionList objects using the '==' symbol.
Method modify_transforms(new_transform_list): Changes the transforms that define the relationship between the space in which points are returned (the untransformed space) and the space in which JumpingDistributions are defined (the transformed space). The argument should be a TransformList of length numparams (or something that can be cast to one using the TransformList.cast(key, numparams) method from the distpy.transform submodule).
Method transformed_version(): Returns a version of this JumpingDistributionList where both the untransformed and transformed space of the new JumpingDistributionList are the transformed space of this JumpingDistributionList.
Property transform_list: TransformList object defining the space in which parameters are returned from the underlying jumping distributions.
Property empty: True if and only if the JumpingDistributionList has no JumpingDistribution objects in it.

Class: JumpingDistributionSet
Signature: JumpingDistributionSet(jumping_distribution_tuples=[])
Description: The JumpingDistributionSet class allows for an unordered set of JumpingDistribution objects which can be defined in transformed space and can be drawn from and evaluated simultaneously. The initialization argument jumping_distribution_tuples should be a list of tuples, tup, that could be individually passed to the add_distribution method (see below) through jumping_distribution_set.add_distribution(*tup). The JumpingDistributionSet class can also be initialized with no arguments, in which case the user can add jumping distributions one at a time using the add_distribution method below. The parameters are referred to through string names because no indexing is implied in inputs and outputs (both are dictionaries).
Method add_distribution(jumping_distribution, params, transforms=None): Adds a jumping distribution to the set, defined in the given transformed space. params should be a sequence of strings of length jumping_distribution.numparams, which is used to define the names of the parameters of this JumpingDistributionSet. When accepting inputs or putting out outputs, this JumpingDistributionSet uses dictionaries whose keys are these strings. transforms should be something that can be cast to a TransformList of length jumping_distribution.numparams using the TransformList.cast function described above in the distpy.transform submodule. If no transforms are given, then the jumping_distribution takes inputs and puts out outputs points that are in the same space in which the corresponding distribution is defined.
Method find_distribution(parameter): When given a string name in the parameter argument, this method returns a tuple of the form (jumping_distribution, index, transform) where jumping_distribution is the JumpingDistribution object describing this parameter, index is the integer number (starting at 0) of the given parameter in the jumping distribution, and transform is the Transform object defining the space in which JumpingDistribution inputs or outputs the parameter.
Method __getitem__(parameter): Magic method alias of find_distribution above that allows for square bracket indexing when searching.
Method delete_distribution(parameter): Deletes the jumping distribution describing the given parameter (this also deletes the distribution of other parameters which share a Distribution object with this one).
Method __delitem__(parameter): Magic method alias of delete_distribution above that allows for square bracket indexing and the use of the del keyword when deleting jumping distributions.
Method __add__(other): Magic method allowing JumpingDistributionSet objects to be combined using the '+' symbol.
Method __len__(): Allows user to check the number of parameters in the JumpingDistributionSet without explicitly referencing the numparams property.
Method __eq__(other): Allows for equality checking with JumpingDistributionSet objects using the '==' symbol.
Method __ne__(other): Allows for inequality checking with JumpingDistributionSet objects using the '!=' symbol. Always returns the opposite of __eq__(other)
Method fill_hdf5_group(group): method that saves the JumpingDistributionSet to an h5py.Group (such as an open hdf5 file or some directory inside it)
Method save(file_name): method that saves the JumpingDistributionSet to a new hdf5 file located at file_name.
Method load_from_hdf5_group(group): Static method that loads and returns a JumpingDistributionSet object from a h5py.Group (such as an open hdf5 file or some directory inside it).
Method load(file_name): Static method that loads and returns a JumpingDistributionSet object from an hdf5 file located at file_name.
Method draw(source, shape=None, random=numpy.random): Draws a random sample from the JumpingDistributionSet when source is a dictionary of source variable values. This function returns a dictionary whose keys are parameter names and whose values are random values of given shape (setting shape to None returns single values as keys). The random keyword argument allows the user to supply a RandomState (the default is to use simply numpy.random).
Method log_value(source, destination): Evaluates the log density of destination when jumping from source (in untransformed space). source and destination should be dictionaries with parameter names as keys and variable values as values.
Method log_value_difference(source, destination): Evaluates (log_value(source, destination) - log_value(destination, source))
Method continuous_subset(): Returns a version of this JumpingDistributionSet where only the continuous parameters/jumping distributions are included.
Method discrete_subset(): Returns a version of this JumpingDistributionSet where only the discrete parameters/jumping distributions are included.
Method jumping_distribution_list(parameters): Returns a JumpingDistributionList including the jumping distributions in the order corresponding to the given parameters.
Method modify_parameters(function): Modifies the names of the parameters of this jumping distribution. The argument should be a Callable which, when called with old parameter names as inputs, returns new parameter names as outputs.
Method modify_transforms(**new_transforms): Changes the transforms that define the relationship between the space in which points are returned (the untransformed space) and the space in which JumpingDistributions are defined (the transformed space). The keyword arguments given to this function should include all parameters as keys and the new Transform objects (or things that can be cast to Transform objects using the cast_to_transform(key) method from the distpy.transform submodule) as values.
Method transformed_version(): Returns a version of this JumpingDistributionSet where both the untransformed space and the transformed space of the new DistributionSet are the transformed space of this JumpingDistributionSet.
Method plot_univariate_histogram(source, ndraw, parameter, in_transformed_space=True, reference_value=None, bins=None, matplotlib_function='fill_between', show_intervals=False, norm_by_max=True, xlabel='', ylabel='', title='', fontsize=28, ax=None, show=False, contour_confidence_levels=0.95, **kwargs): Plots a univariate histogram of the given parameter assuming that the JumpingDistributionSet is jumping from source using the univariate_histogram function of the distpy.util submodule. See the description of that function for documentation on the keyword_arguments. ndraw should be an integer number of samples to draw to make the histogram.
Method plot_bivariate_histogram(source, ndraw, parameter1, parameter2, in_transformed_space=True, reference_value_mean=None, reference_value_covariance=None, bins=None, matplotlib_function='imshow', xlabel='', ylabel='', title='', fontsize=28, ax=None, show=False, contour_confidence_levels=0.95, **kwargs): Plots a bivariate histogram of the given parameters assuming that the JumpingDistributionSet is jumping from source using the bivariate_histogram function of the distpy.util submodule. See the description of that function for documentation on the keyword arguments. ndraw should be an integer number of samples to draw to make the histogram.
Method triangle_plot(source, ndraw, parameters=None, in_transformed_space=True, figsize=(8, 8), fig=None, show=False, kwargs_1D={}, kwargs_2D={}, fontsize=28, nbins=100, plot_type='contour', reference_value_mean=None, reference_value_covariance=None, contour_confidence_levels=0.95, parameter_renamer=(lambda x: x), tick_label_format_string='{x:.3g}'): Makes a triangle plot of the given parameters (None means all parameters are plotted) assuming that the JumpingDistributionSet is jumping from source using the triangle_plot function from the distpy.util submodule. See the description of that function for documentation of the keyword arguments. ndraw should be an integer number of samples to draw to make the histograms.
Property params: Sequence of strings representing the parameters described by this jumping distribution in an internal order.
Property numparams: Integer number of parameters described by the JumpingDistributionSet.
Property continuous_params: Returns a list of strings representing the continuous parameters described by this jumping distribution.
Property discrete_params: Returns a list of strings representing the discrete parameters described by this jumping distribution.
Property empty: True if and only if the DistributionSet has no Distribution objects in it.
Property transform_set: TransformSet object yielding the Transform objects corresponding to eahc parameter name.


The distpy.jumping submodule provides the following methods to load distributions from hdf5 files or groups.

Function: load_jumping_distribution_from_hdf5_group
Signature: load_jumping_distribution_from_hdf5_group(group, *args)
Description: Loads a JumpingDistribution object from the given h5py.Group (such as an open hdf5 file or a subdirectory of it), where it was earlier saved using the fill_hdf5_group(group) method of the JumpingDistribution class.

Function: load_jumping_distribution_from_hdf5_file
Signature: load_jumping_distribution_from_hdf5_file(file_name)
Description: Loads a JumpingDistribution object from an existing hdf5 file at the given location.


The distpy.jumping submodule also provides the MetropolisHastingsSampler class, a conceptually simple implementation of the Metropolis Hastings Markov Chain Monte Carlo sampler:

Class: MetropolisHastingsSampler
Signature: MetropolisHastingsSampler(parameters, nwalkers, logprobability,\
        jumping_distribution_set, nthreads=1, args=[], kwargs={})
Description: A simple MetropolisHastings sampler with an arbitrary JumpingDistributionSet proposal distribution jumping_distribution_set. parameters should be a list of strings. nwalkers should be a positive integer indicating the number of independent iterates to evolve. args and kwargs are positional and keyword arguments passed to logprobability, which should be a callable that, when given the position of a current iterate along with args and kwargs, returns the log density of the posterior distribution to explore. nthreads should be a positive integer determining the number of threads to use in calculating the logprobability many times. This generally only provides a benefit if logprobability requires more than a few tens of ms to evaluate.
Method sample(point, lnprob=None, randomstate=None, thin=1, storechain=True, iterations=1): A generator that, at each evaluation, yields (pos, lnprob, rstate) where pos is the current positions of the chain in the parameter space, lnprob is the value of the log posterior at pos, and rstate is the current state of the random number generator. point should be a 2D numpy.ndarray of shape (nwalkers, numparams), lnprob (if given) should be a 1D numpy.ndarray of shape (nwalkers,), and rstate should be a state of a random number generator. thin allows user to control how often points should be saved to memory. storechain allows user to control whether chain should be saved to memory as samples are generated or if the user will do that themself. iterations determines the number of steps each walker should take upon each evaluation.
Method reset(): Resets the sampler, clearing chain, lnprob, and other properties.

