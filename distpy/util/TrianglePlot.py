"""
File: distpy/util/TrianglePlot.py
Author: Keith Tauscher
Date: 18 Aug 2018

Description: File containing functions which plot univariate histograms,
             bivariate histograms, and triangle plots (which are really just
             combinations of the previous two types).
"""
from __future__ import division
import numpy as np
import scipy.linalg as scila
from .TypeCategories import real_numerical_types, sequence_types

try:
    import matplotlib.pyplot as pl
    from matplotlib.ticker import StrMethodFormatter
except:
    have_matplotlib = False
else:
    have_matplotlib = True
no_matplotlib_error = ImportError("matplotlib cannot be imported.")

def univariate_histogram(sample, reference_value=None, bins=None,\
    matplotlib_function='fill_between', show_intervals=False, xlabel='',\
    ylabel='', title='', fontsize=28, ax=None, show=False, norm_by_max=True,\
    **kwargs):
    """
    Plots a 1D histogram of the given sample.
    
    sample: the 1D sample of which to take a histogram
    reference_value: a point at which to plot a dashed reference line
    bins: bins to pass to numpy.histogram: default, None
    matplotlib_function: either 'fill_between', 'bar', or 'plot'
    show_intervals: if True, 95% confidence intervals are plotted
    xlabel: the string to use in labeling x axis
    ylabel: the string to use in labeling y axis
    title: title string with which to top plot
    fontsize: the size of the tick label font
    ax: if None, new Figure and Axes are created
        otherwise, this Axes object is plotted on
    show: if True, matplotlib.pyplot.show is called before this function
                   returns
    norm_by_max: if True, normalization is such that maximum of histogram
                          values is 1. Default: True
    kwargs: keyword arguments to pass on to matplotlib.Axes.plot or
            matplotlib.Axes.fill_between
    
    returns: None if show is True, otherwise Axes instance with plot
    """
    if not have_matplotlib:
        raise no_matplotlib_error
    if type(ax) is type(None):
        fig = pl.figure()
        ax = fig.add_subplot(111)
    (nums, bins) = np.histogram(sample, bins=bins)
    bin_centers = (bins[1:] + bins[:-1]) / 2
    num_bins = len(bin_centers)
    if norm_by_max:
        nums = nums / np.max(nums)
    ylim = (0, 1.1 * np.max(nums))
    if 'color' in kwargs:
        color = kwargs['color']
        del kwargs['color']
    else:
        # 95% interval color
        color = 'C0'
    cumulative = np.cumsum(nums)
    cumulative = cumulative / cumulative[-1]
    cumulative_is_less_than_025 = np.argmax(cumulative > 0.025)
    cumulative_is_more_than_975 = np.argmax(cumulative > 0.975) + 1
    interval_95p =\
        (cumulative_is_less_than_025, cumulative_is_more_than_975 + 1)
    if matplotlib_function in ['bar', 'plot']:
        if matplotlib_function == 'bar':
            ax.bar(bin_centers, nums,\
                width=(bins[-1] - bins[0]) / num_bins, color=color, **kwargs)
        else:
            ax.plot(bin_centers, nums, color=color, **kwargs)
        if show_intervals:
            ax.plot([bins[interval_95p[0]]]*2, ylim, color='r', linestyle='--')
            ax.plot([bins[interval_95p[1]]]*2, ylim, color='r', linestyle='--')
    elif matplotlib_function == 'fill_between':
        if show_intervals:
            ax.plot(bin_centers, nums, color='k', linewidth=1)
            half_bins = np.linspace(bins[0], bins[-1], (2 * len(bins)) - 1)
            interpolated_nums = np.interp(half_bins, bin_centers, nums)
            ax.fill_between(\
                half_bins[2*interval_95p[0]:2*interval_95p[1]],\
                np.zeros((2 * (interval_95p[1] - interval_95p[0]),)),\
                interpolated_nums[2*interval_95p[0]:2*interval_95p[1]],\
                color=color)
            ax.fill_between(bin_centers, nums,\
                np.ones_like(nums) * 1.5 * np.max(nums), color='w')
        else:
            ax.fill_between(bin_centers, np.zeros_like(nums), nums,\
                color=color, **kwargs)
    else:
        raise ValueError("matplotlib_function not recognized.")
    ax.set_ylim(ylim)
    if type(reference_value) is not type(None):
        ax.plot([reference_value] * 2, ylim, color='r', linewidth=1,\
            linestyle='--')
        ax.set_ylim(ylim)
    ax.set_xlim((bins[0], bins[-1]))
    ax.set_xlabel(xlabel, size=fontsize)
    ax.set_ylabel(ylabel, size=fontsize)
    ax.set_title(title, size=fontsize)
    ax.tick_params(width=2, length=6, labelsize=fontsize)
    if show:
        pl.show()
    else:
        return ax

def confidence_contour_2D(xsample, ysample, nums=None,\
    confidence_contours=0.95, hist_kwargs={}):
    """
    Finds the posterior distribution levels which represent the boundaries of
    confidence intervals of the given confidence level(s).
    
    xsample: xsample to contour
    ysample: ysample to contour
    nums: if histogram has already been created, nums can be passed here
    confidence_contours: confidence level as a number between 0 and 1 or a 1D
                         array of such numbers, default: 0.95
    hist_kwargs: only used if nums is None, contains keyword arguments to pass
                 to histogram2d function
    
    returns: 1D array of confidence contours
    """
    if type(nums) is type(None):
        (nums, xedges, yedges) =\
            np.histogram2d(xsample, ysample, **hist_kwargs)
    nums = np.sort(nums.flatten())
    cdf_values = np.cumsum(nums)
    cdf_values = (cdf_values / cdf_values[-1])
    confidence_levels = 1 - cdf_values
    if type(confidence_contours) in real_numerical_types:
        confidence_contours = [confidence_contours]
    if type(confidence_contours) in sequence_types:
        confidence_contours = np.sort(confidence_contours)
        return np.where(np.all(confidence_levels[np.newaxis,:] <=\
            confidence_contours[:,np.newaxis], axis=-1), nums[0],\
            np.interp(confidence_contours, confidence_levels[-1::-1],\
            nums[-1::-1]))
    else:
        raise TypeError("confidence_contours was set to neither a single " +\
            "number or a 1D sequence of numbers.")

def bivariate_histogram(xsample, ysample, reference_value_mean=None,\
    reference_value_covariance=None, bins=None, matplotlib_function='imshow',\
    xlabel='', ylabel='', title='', fontsize=28, ax=None, show=False,\
    contour_confidence_levels=0.95, reference_color='r', reference_alpha=1,\
    minima=None, maxima=None, num_ellipse_points=1000, **kwargs):
    """
    Plots a 2D histogram of the given joint sample.
    
    xsample: the sample to use for the x coordinates
    ysample: the sample to use for the y coordinates
    reference_value_mean: points to plot a dashed reference line for axes
    reference_value_covariance: if not None, used (along with
                                reference_value_mean) to plot reference ellipse
    bins: bins to pass to numpy.histogram2d, default: None
    matplotlib_function: function to use in plotting. One of ['imshow',
                         'contour', 'contourf']. default: 'imshow'
    xlabel: the string to use in labeling x axis
    ylabel: the string to use in labeling y axis
    title: title with which to top plot
    fontsize: the size of the tick label font (and other fonts)
    ax: if None, new Figure and Axes are created
        otherwise, this Axes object is plotted on
    show: if True, matplotlib.pyplot.show is called before this function
                   returns
    contour_confidence_levels: the confidence level of the contour in the
                               bivariate histograms. Only used if
                               matplotlib_function is 'contour' or 'contourf'
                               or if reference_value_mean and
                               reference_value_covariance are both not None.
                               Can be single number or sequence of numbers
    minima: sequence of (min_X, min_Y) to take into account when plotting
            ellipses (only used if reference_value_covariance is not None)
    maxima: sequence of (max_X, max_Y) to take into account when plotting
            ellipses (only used if reference_value_covariance is not None)
    kwargs: keyword arguments to pass on to matplotlib.Axes.imshow (any but
            'origin', 'extent', or 'aspect') or matplotlib.Axes.contour or
            matplotlib.Axes.contourf (any)
    
    returns: None if show is True, otherwise Axes instance with plot
    """
    if not have_matplotlib:
        raise no_matplotlib_error
    if type(ax) is type(None):
        fig = pl.figure()
        ax = fig.add_subplot(111)
    (nums, xbins, ybins) = np.histogram2d(xsample, ysample, bins=bins)
    if matplotlib_function == 'contour':
        nums = np.concatenate([np.zeros((1, nums.shape[1])), nums,\
            np.zeros((1, nums.shape[1]))], axis=0)
        nums = np.concatenate([np.zeros((nums.shape[0], 1)), nums,\
            np.zeros((nums.shape[0], 1))], axis=1)
        xbins = np.concatenate([[(2 * xbins[0]) - xbins[1]], xbins,\
            [(2 * xbins[-1]) - xbins[-2]]])
        ybins = np.concatenate([[(2 * ybins[0]) - ybins[1]], ybins,\
            [(2 * ybins[-1]) - ybins[-2]]])
    xlim = (xbins[0], xbins[-1])
    ylim = (ybins[0], ybins[-1])
    xbin_centers = (xbins[1:] + xbins[:-1]) / 2
    ybin_centers = (ybins[1:] + ybins[:-1]) / 2
    if matplotlib_function == 'imshow':
        ax.imshow(nums.T, origin='lower',\
            extent=[xlim[0], xlim[1], ylim[0], ylim[1]], aspect='auto',\
            **kwargs)
    else:
        pdf_max = np.max(nums)
        contour_levels = confidence_contour_2D(xsample, ysample, nums=nums,\
            confidence_contours=contour_confidence_levels)
        contour_levels = np.sort(contour_levels)
        if matplotlib_function == 'contour':
            ax.contour(xbin_centers, ybin_centers, nums.T, contour_levels,\
                **kwargs)
        elif matplotlib_function == 'contourf':
            contour_levels = np.concatenate([contour_levels, [pdf_max]])
            ax.contourf(xbin_centers, ybin_centers, nums.T, contour_levels,\
                **kwargs)
        else:
            raise ValueError("matplotlib_function not recognized.")
    if type(reference_value_mean) is not type(None):
        if type(reference_value_mean[0]) is not type(None):
            ax.plot([reference_value_mean[0]] * 2, ylim,\
                color=reference_color, linewidth=1, linestyle='--')
        if type(reference_value_mean[1]) is not type(None):
            ax.plot(xlim, [reference_value_mean[1]] * 2,\
                color=reference_color, linewidth=1, linestyle='--')
        if (type(reference_value_mean[0]) is not type(None)) and\
            (type(reference_value_mean[1]) is not type(None)) and\
            (type(reference_value_covariance) is not type(None)):
            reference_value_mean = np.array(reference_value_mean)
            if type(contour_confidence_levels) in real_numerical_types:
                contour_level_for_2D_reference_contour =\
                    contour_confidence_levels
            elif type(contour_confidence_levels) in sequence_types:
                contour_level_for_2D_reference_contour =\
                    np.max(contour_confidence_levels)
            else:
                raise ValueError("For reference value contours to be " +\
                    "plotted, contour_confidence_levels should be either a " +\
                    "single number between 0 or 1 or a sequence of such " +\
                    "numbers.")
            covariance_expansion_factor_for_2D_reference_contour =\
                ((-2) * np.log(1 - contour_level_for_2D_reference_contour))
            sqrt_covariance_matrix = scila.sqrtm(reference_value_covariance) *\
                np.sqrt(covariance_expansion_factor_for_2D_reference_contour)
            angles = np.linspace(0, 2 * np.pi, num=num_ellipse_points,\
                endpoint=False)
            circle_points = np.array([np.cos(angles), np.sin(angles)])
            ellipse_points = reference_value_mean[:,np.newaxis] +\
                np.dot(sqrt_covariance_matrix, circle_points)
            (ellipse_xs, ellipse_ys) = ellipse_points
            if type(minima) is type(None):
                minima = (-np.inf, -np.inf)
            if type(maxima) is type(None):
                maxima = (+np.inf, +np.inf)
            ellipse_xs =\
                np.where(ellipse_xs < minima[0], minima[0], ellipse_xs)
            ellipse_xs =\
                np.where(ellipse_xs > maxima[0], maxima[0], ellipse_xs)
            ellipse_ys =\
                np.where(ellipse_ys < minima[1], minima[1], ellipse_ys)
            ellipse_ys =\
                np.where(ellipse_ys > maxima[1], maxima[1], ellipse_ys)
            ax.fill(ellipse_xs, ellipse_ys,\
                linewidth=(1 if (matplotlib_function=='contour') else 0),\
                fill=(matplotlib_function=='contourf'), linestyle='--',\
                color=reference_color, alpha=reference_alpha)
    ax.tick_params(width=2.5, length=7.5, labelsize=fontsize)
    ax.set_xlabel(xlabel, size=fontsize)
    ax.set_ylabel(ylabel, size=fontsize)
    ax.set_title(title, size=fontsize)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if show:
        pl.show()
    else:
        return ax

def get_ax_with_geometry(fig, *geometry):
    """
    Gets the Axes with the given geometry.
    
    fig: Matplotlib Figure object
    geometry: row, column, and plot index (starting at 1) in tuple
    
    returns: a Matplotlib Axes object
    """
    for ax in fig.axes:
        if ax.get_geometry() == tuple(geometry):
            return ax
    raise KeyError("No plot has the given geometry.")

def triangle_plot(samples, labels, figsize=(8, 8), fig=None, show=False,\
    kwargs_1D={}, kwargs_2D={}, fontsize=28, nbins=100,\
    plot_type='contour', reference_value_mean=None,\
    reference_value_covariance=None, contour_confidence_levels=0.95,\
    minima=None, maxima=None, tick_label_format_string='{x:.3g}', num_ticks=3,\
    minor_ticks_per_major_tick=1, xlabel_rotation=0, xlabelpad=None,\
    ylabel_rotation=90, ylabelpad=None):
    """
    Makes a triangle plot out of N samples corresponding to (possibly
    correlated) random variables
    
    samples: tuple of N 1D samples of the same length or an array of shape
             (N,m) where m is a single integer
    labels: the labels to use for each sample
    figsize: the size of the figure on which to put the triangle plot
    show: if True, matplotlib.pyplot.show is called before this function
                   returns
    kwargs_1D: keyword arguments to pass on to univariate_histogram function
    kwargs_2D: keyword arguments to pass on to bivariate_histogram function
    fontsize: the size of the label fonts
    nbins: the number of bins for each sample
    plot_type: determines the matplotlib functions to use for univariate and
               bivariate histogram
               if plot_type=='contourf': 'bar' and 'contourf' are used
               if plot_type=='contour': 'plot' and 'contour' are used
               if plot_type=='histogram': 'bar' and 'imshow' are used
               otherwise: plot_type should be a length-2 sequence of the form
                          (matplotlib_function_1D, matplotlib_function_2D)
    reference_value_mean: reference values to place on plots, if there are any
    reference_value_covariance: if not None, used (along with
                                reference_value_mean) to plot reference
                                ellipses in each bivariate histogram
    contour_confidence_levels: the confidence level of the contour in the
                               bivariate histograms. Only used if plot_type is
                               'contour' or 'contourf'. Can be single number or
                               sequence of numbers
    minima: sequence of variable minima to take into account when plotting
            ellipses (only used if reference_value_covariance is not None)
    maxima: sequence of variable maxima to take into account when plotting
            ellipses (only used if reference_value_covariance is not None)
    tick_label_format_string: format string that can be called using
                              tick_label_format_string.format(x=loc) where loc
                              is the location of the tick in data coordinates
    num_ticks: integer number of ticks per panel, default: 3
    minor_ticks_per_major_tick: integer number of minor ticks per major tick
    xlabel_rotation: rotation of x-label in degrees, default: 0
    xlabelpad: pad size for xlabel, default: None
    ylabel_rotation: rotation of y-label in degrees, default: 90
    ylabelpad: pad size for ylabel, default: None
    
    returns: None if show is True, otherwise Figure instance with plot
    """
    if not have_matplotlib:
        raise no_matplotlib_error
    if type(fig) is type(None):
        fig = pl.figure(figsize=figsize)
    existing_plots = bool(fig.axes)
    samples = np.array(samples)
    num_samples = samples.shape[0]
    if type(minima) is type(None):
        minima = np.array([-np.inf] * num_samples)
    if type(maxima) is type(None):
        maxima = np.array([+np.inf] * num_samples)
    (full_kwargs_1D, full_kwargs_2D) = ({}, {})
    if plot_type == 'contour':
        matplotlib_function_1D = 'plot'
        matplotlib_function_2D = 'contour'
        if 'colors' not in kwargs_2D:
            full_kwargs_2D['cmap'] = 'Dark2'
    elif plot_type == 'contourf':
        matplotlib_function_1D = 'bar'
        matplotlib_function_2D = 'contourf'
        full_kwargs_2D['colors'] =\
            ['C{:d}'.format(index) for index in [0, 2, 4, 6, 1, 3, 5, 7]]
    elif plot_type == 'histogram':
        matplotlib_function_1D = 'bar'
        matplotlib_function_2D = 'imshow'
    elif (type(plot_type) in sequence_types) and (len(plot_type) == 2):
        (matplotlib_function_1D, matplotlib_function_2D) = plot_type
    else:
        raise ValueError("plot_type not recognized.")
    full_kwargs_1D.update(kwargs_1D)
    full_kwargs_2D.update(kwargs_2D)
    (ticks, minor_ticks) = ([], [])
    bins = []
    for (isample, sample) in enumerate(samples):
        min_to_include = np.min(sample)
        max_to_include = np.max(sample)
        if (type(reference_value_mean) is not type(None)) and\
            (type(reference_value_mean[isample]) is not type(None)):
            min_to_include =\
                min(min_to_include, reference_value_mean[isample])
            max_to_include =\
                max(max_to_include, reference_value_mean[isample])
        middle = (max_to_include + min_to_include) / 2
        half_width = (max_to_include - min_to_include) / 2
        these_bins = np.linspace(min_to_include - (half_width / 5),\
            max_to_include + (half_width / 5), nbins + 1)
        bins.append(these_bins)
        half_width = (these_bins[-1] - these_bins[0]) / 2
        major_tick_low_endpoint = middle - (half_width * (1 - (1 / num_ticks)))
        major_tick_high_endpoint =\
            middle + (half_width * (1 - (1 / num_ticks)))
        these_ticks = np.linspace(major_tick_low_endpoint,\
            major_tick_high_endpoint, num_ticks)
        ticks.append(these_ticks)
        minor_tick_low_endpoint = middle - (half_width * (1 + (1 / num_ticks)))
        minor_tick_high_endpoint =\
            middle + (half_width * (1 + (1 / num_ticks)))
        num_minor_ticks =\
            ((num_ticks + 1) * (minor_ticks_per_major_tick + 1)) + 1
        these_minor_ticks = np.linspace(minor_tick_low_endpoint,\
            minor_tick_high_endpoint, num_minor_ticks)
        minor_ticks.append(these_minor_ticks)
    tick_label_formatter = StrMethodFormatter(tick_label_format_string)
    for (column, column_sample) in enumerate(samples):
        column_label = labels[column]
        if type(reference_value_mean) is type(None):
            reference_value_x = None
        else:
            reference_value_x = reference_value_mean[column]
        for (row, row_sample) in enumerate(samples):
            if row < column:
                continue
            row_label = labels[row]
            plot_number = ((num_samples * row) + column + 1)
            if existing_plots:
                ax = get_ax_with_geometry(fig, num_samples, num_samples,\
                    plot_number)
            else:
                ax = fig.add_subplot(num_samples, num_samples, plot_number)
            if row == column:
                univariate_histogram(column_sample,\
                    reference_value=reference_value_x,\
                    bins=bins[column],\
                    matplotlib_function=matplotlib_function_1D,\
                    show_intervals=False, xlabel='', ylabel='', title='',\
                    fontsize=fontsize, ax=ax, show=False, **full_kwargs_1D)
            else:
                if type(reference_value_mean) is type(None):
                    reference_value_y = None
                else:
                    reference_value_y = reference_value_mean[row]
                reference_value_submean =\
                    (reference_value_x, reference_value_y)
                if type(reference_value_covariance) is type(None):
                    reference_value_subcovariance = None
                else:
                    indices = np.array([column, row])
                    reference_value_subcovariance =\
                        reference_value_covariance[indices,:][:,indices]
                these_minima = (minima[column], minima[row])
                these_maxima = (maxima[column], maxima[row])
                bivariate_histogram(column_sample, row_sample,\
                    reference_value_mean=reference_value_submean,\
                    reference_value_covariance=reference_value_subcovariance,\
                    bins=(bins[column], bins[row]),\
                    matplotlib_function=matplotlib_function_2D, xlabel='',\
                    ylabel='', title='', fontsize=fontsize, ax=ax,\
                    show=False,\
                    contour_confidence_levels=contour_confidence_levels,\
                    minima=these_minima, maxima=these_maxima, **full_kwargs_2D)
            ax.set_xticks(minor_ticks[column], minor=True)
            ax.set_xticks(ticks[column], minor=False)
            if row != column:
                ax.set_yticks(minor_ticks[row], minor=True)
                ax.set_yticks(ticks[row], minor=False)
            ax.xaxis.set_major_formatter(tick_label_formatter)
            ax.yaxis.set_major_formatter(tick_label_formatter)
            ax.tick_params(width=2.5, length=7.5, which='major')
            ax.tick_params(width=1.5, length=4.5, which='minor')
            ax.tick_params(left=True, right=True, top=True, bottom=True,\
                labelleft=False, labelright=False, labeltop=False,\
                labelbottom=False, direction='inout', which='both')
            if (row == column):
                ax.tick_params(left=False, top=False, right=False,\
                    which='both')
            if (row + 1) == num_samples:
                ax.set_xlabel(column_label, size=fontsize,\
                    rotation=xlabel_rotation, labelpad=xlabelpad)
                ax.tick_params(labelbottom=True, which='major')
            if column == 0:
                if row == 0:
                    ax.tick_params(labelleft=False)
                else:
                    ax.set_ylabel(row_label, size=fontsize,\
                        rotation=ylabel_rotation, labelpad=ylabelpad)
                    ax.tick_params(labelleft=True, which='major')
            ax.set_xlim((bins[column][0], bins[column][-1]))
            if row != column:
                ax.set_ylim((bins[row][0], bins[row][-1]))
    fig.subplots_adjust(wspace=0, hspace=0)
    if show:
        pl.show()
    else:
        return fig

