#!/bin/bash

for FILENAME in $(ls $(dirname $(realpath $0))/*.py) ; do python $FILENAME ; done
