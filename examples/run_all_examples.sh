#!/bin/bash

for fname in $(ls $DISTPY/examples/*.py) ; do ipython $fname ; done
