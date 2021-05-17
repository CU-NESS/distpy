"""
Introduces utilities used throughout the package, including:

- interfaces for making objects `distpy.util.Savable.Savable` and
  `distpy.util.Loadable.Loadable` in binary hdf5 files using h5py
- helper methods for using h5py to save and load variables and arrays
(`h5py_extensions`)  
- type category definitions (`distpy.util.TypeCategories`)  
- functions for making univariate histograms, bivariate histograms, and
  triangle plots (`distpy.util.TrianglePlot`)  
- a class that uses strings to represent an `distpy.util.Expression.Expression`
  that can be modified and have arguments passed to it before being evaluated

**File**: $DISTPY/distpy/util/\\_\\_init\\_\\_.py  
**Author**: Keith Tauscher  
**Date**: 14 May 2021
"""
from distpy.util.Savable import Savable
from distpy.util.Loadable import Loadable
from distpy.util.TypeCategories import bool_types, int_types, float_types,\
    real_numerical_types, complex_numerical_types, numerical_types,\
    sequence_types
from distpy.util.h5py_extensions import create_hdf5_dataset, get_hdf5_value,\
    HDF5Link, save_dictionary, load_dictionary
from distpy.util.TrianglePlot import univariate_histogram,\
    confidence_contour_2D, bivariate_histogram, triangle_plot
from distpy.util.Expression import Expression

