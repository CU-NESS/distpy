#!/usr/bin/env python

import os, urllib

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

packages = ['distpy']
        
setup(name='distpy',
      version='0.1',
      description='Distributions in Python',
      packages=packages,
     )

DISTPY_env = os.getenv('DISTPY')
cwd = os.getcwd()

##
# TELL PEOPLE TO SET ENVIRONMENT VARIABLE
##
if not DISTPY_env:

    import re    
    shell = os.getenv('SHELL')

    print "\n"
    print "#"*78
    print "It would be in your best interest to set an environment variable"
    print "pointing to this directory.\n"

    if shell:    

        if re.search('bash', shell):
            print "Looks like you're using bash, so add the following to " +\
                  "your .bashrc:"
            print "\n    export DISTPY=%s" % cwd
        elif re.search('csh', shell):
            print "Looks like you're using csh, so add the following to " +\
                  "your .cshrc:"
            print "\n    setenv DISTPY %s" % cwd        

    print "\nGood luck!"
    print "#"*78        
    print "\n"

# Print a warning if there's already an environment variable but it's pointing
# somewhere other than the current directory
elif DISTPY_env != cwd:

    print "\n"
    print "#"*78
    print "It looks like you've already got a DISTPY environment variable set",
    print "but it's \npointing to a different directory:"
    print "\n    DISTPY=%s" % DISTPY_env

    print "\nHowever, we're currently in %s.\n" % cwd

    print "Is this a different install (might not cause problems),",
    print "or perhaps just"
    print "a typo in your environment variable?"

    print "#"*78        
    print "\n"
    
    
    
