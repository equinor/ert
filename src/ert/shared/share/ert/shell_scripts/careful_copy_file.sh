#! /usr/bin/env bash

if [ -f $1 ]; then
    if [ -f $2 ]; then
        echo "File ${2} already present - not updated" >&2
    else
        mkdir -p $2
        cp $1 $2
    fi
else
    echo "Input argument ${1} does not correspond to an existing file" >&2
    exit 1
fi
