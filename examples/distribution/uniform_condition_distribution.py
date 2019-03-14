"""
File: examples/distribution/uniform_condition_distribution.py
Author: Keith Tauscher
Date: 14 Mar 2019

Description: Example script showing how to use the
             UniformConditionDistribution class. In this case, a simple
             UniformDistribution is recreated with the
             UniformConditionDistribution class.
"""
import os
import numpy as np
import matplotlib.pyplot as pl
from distpy import Expression, UniformConditionDistribution

(low, high) = (-27., 19.)

expression_string = '(({{0}} > {0}) and ({{0}} < {1}))'.format(low, high)
print("expression_string='{!s}'".format(expression_string))
expression = Expression(expression_string)
distribution = UniformConditionDistribution(expression)
hdf5_file_name = 'TEST_DELETE_THIS.hdf5'
distribution.save(hdf5_file_name)
try:
    assert (distribution == UniformConditionDistribution.load(hdf5_file_name))
except:
    os.remove(hdf5_file_name)
    raise
else:
    os.remove(hdf5_file_name)
assert(distribution.numparams == 1)
fig = pl.figure()
ax = fig.add_subplot(111)
xs = np.linspace(-30, 20, 5001)
distribution.plot(xs, ax=ax, show=False, linewidth=2, color='r',\
    label='e^(log_value)')
ax.set_title(('Uniform condition distribution on [{0!s},{1!s}]').format(low,\
    high), size='xx-large')
ax.set_xlabel('Value', size='xx-large')
ax.set_ylabel('PDF', size='xx-large')
ax.tick_params(labelsize='xx-large', width=2, length=6)
ax.legend(fontsize='xx-large', loc='lower center')
pl.show()

