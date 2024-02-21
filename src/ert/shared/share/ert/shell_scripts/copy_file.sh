#! /usr/bin/env bash

if [ -f $1 ]; then
    mkdir -p $2
    echo "Copying file '${1}' -> '${2}'"
    cp $1 $2
else
    echo "Input argument ${1} does not correspond to an existing file" >&2
    exit 1
fi
