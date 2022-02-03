#!/bin/bash
set -euo pipefail

case "$1" in
    3.6) pyver=cp36-cp36m ;;
    3.7) pyver=cp37-cp37m ;;
    3.8) pyver=cp38-cp38 ;;
    3.9) pyver=cp39-cp39 ;;
    3.10) pyver=cp310-cp310 ;;
    *)
        echo "Unknown Python version $1"
        exit 1
        ;;
esac

# Build wheel
cd /github/workspace
/opt/python/$pyver/bin/pip wheel . --no-deps -w wheelhouse
auditwheel repair wheelhouse/* -w dist
