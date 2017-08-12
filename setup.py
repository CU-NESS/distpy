#!/usr/bin/env python
"""
File: setup.py
Author: Keith Tauscher
Date: 12 Aug 2017

Description: Installs distpy.
"""
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
        
setup(name='distpy', version='0.1', description='Distributions in Python',\
    packages=['distpy'])

    
    
    
