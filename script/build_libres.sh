#!/bin/bash

set -e
set -x

rm -rf ../build
mkdir ../build
pushd ../build

cmake ../libres -DBUILD_TESTS=ON
cmake --build .
