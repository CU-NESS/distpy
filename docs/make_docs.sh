#!/bin/bash

if [ -n "$DISTPY" ]
then
    cd $DISTPY/docs
    pdoc --config latex_math=True --html ../distpy --force
    cd - > /dev/null
else
    echo "DISTPY environment variable must be set for the make_docs.sh script to be used to make the documentation."
fi
