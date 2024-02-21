#! /usr/bin/env bash
if [ -d $1 ]; then
    mkdir -p $2
    echo "Copying directory structure '${1}' -> '${2}'"
    cp -au $1 $2
else
    echo "Input argument ${1} does not correspond to an existing directory" >&2
    exit 1
fi
