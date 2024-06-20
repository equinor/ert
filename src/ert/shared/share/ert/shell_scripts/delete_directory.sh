#! /usr/bin/env bash

if [ -d $1 ]; then
        rm -rf $1
else
    echo "Input argument ${1} does not correspond to an existing directory" >&2
    exit 1
fi
