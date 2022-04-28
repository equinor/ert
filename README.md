# ert

[![Build Status](https://github.com/equinor/ert/actions/workflows/build.yml/badge.svg)](https://github.com/equinor/ert/actions/workflows/build.yml)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ert)](https://img.shields.io/pypi/pyversions/ert)
[![Downloads](https://pepy.tech/badge/ert)](https://pepy.tech/project/ert)
[![GitHub commit activity](https://img.shields.io/github/commit-activity/m/equinor/ert)](https://img.shields.io/github/commit-activity/m/equinor/ert)
[![GitHub contributors](https://img.shields.io/github/contributors-anon/equinor/ert)](https://img.shields.io/github/contributors-anon/equinor/ert)
[![Code Style](https://github.com/equinor/ert/actions/workflows/style.yml/badge.svg)](https://github.com/equinor/ert/actions/workflows/style.yml)
[![Type checking](https://github.com/equinor/ert/actions/workflows/typing.yml/badge.svg)](https://github.com/equinor/ert/actions/workflows/typing.yml)
[![codecov](https://codecov.io/gh/equinor/ert/branch/add_code_coverage/graph/badge.svg?token=keVAcWavZ1)](https://codecov.io/gh/equinor/ert)
[![Run test-data](https://github.com/equinor/ert/actions/workflows/run_ert2_test_data_setups.yml/badge.svg)](https://github.com/equinor/ert/actions/workflows/run_ert2_test_data_setups.yml)
[![Run polynomial demo](https://github.com/equinor/ert/actions/workflows/run_examples_polynomial.yml/badge.svg)](https://github.com/equinor/ert/actions/workflows/run_examples_polynomial.yml)
[![Run SPE1 demo](https://github.com/equinor/ert/actions/workflows/run_examples_spe1.yml/badge.svg)](https://github.com/equinor/ert/actions/workflows/run_examples_spe1.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

ERT - Ensemble based Reservoir Tool - is a tool to run ensemble based on
reservoir models. ERT was originally devised as tool to do model updating
(history matching) with the EnKF method, now the primary method for model
updating is the Ensemble Smoother (ES).

## Prerequisites

Python 3.6+ with development headers.

## Installation

``` sh
$ pip install ert
$ ert --help
```

or, for the latest development version:

``` sh
$ pip install git+https://github.com/equinor/ert.git@master
$ ert --help
```


The `ert` program is based on two different repositories:

1. [ecl](https://github.com/Equinor/ecl) which contains utilities to read and write Eclipse files.

2. ert - this repository - the actual application and all of the GUI.


ERT is now Python 3 only. The last Python 2 compatible release is [2.14](https://github.com/equinor/ert/tree/version-2.14)

## Documentation

Documentation for ert is located at [https://ert.readthedocs.io/en/latest/](https://ert.readthedocs.io/en/latest/).


## Developing

*ERT* uses Python for user-facing code and C++ for some backend code. Python is
the easiest to work with and is likely what most developers will work with.

### Developing Python

To start developing the Python code, install ERT in editable mode into a virtualenv:

```sh
# Create and enable a virtualenv
python3 -m venv my_virtualenv
source my_virtualenv/bin/activate

# Update build dependencies
pip install --upgrade pip wheel setuptools

# Download and install ERT
git clone https://github.com/equinor/ert
cd ert
pip install --editable .
```

Additional development packages must be installed to run the test suite:
```sh
pip install -r dev-requirements.txt
pytest tests/
```

Since we've installed ERT in `--editable` mode, it is now possible to modify the
Python code without reinstalling. Simply restart ERT after making your changes.

### Developing C++

C++ is the backbone of ERT 2 as in used extensively in important parts of ERT.
There's a combination of legacy code and newer refactored code. The end goal is
likely that some core performance-critical functionality will be implemented in
C++ and the rest of the business logic will be implemented in Python.

While running `--editable` will create the necessary Python extension module
(`res/_lib.cpython-*.so`), changing C++ code will not take effect even when
reloading ERT. This requires recompilation, which means reinstalling ERT from
scratch.

To avoid recompiling already-compiled source files, we provide the
`script/build` script. From a fresh virtualenv:

```sh
git clone https://github.com/equinor/ert
cd ert
script/build
```

This command will update `pip` if necessary, install the build dependencies,
compile ERT and install in editable mode, and finally install the runtime
requirements. Further invocations will only build the necessary source files. To
do a full rebuild, delete the `_skbuild` directory.

Note: This will create a debug build, which is faster to compile and comes with
debugging functionality enabled. This means that, for example, Eigen
computations will be checked and will abort if preconditions aren't met (eg.
when inverting a matrix, it will first check that the matrix is square). The
downside is that this makes the code unoptimised and slow. Debugging flags are
therefore not present in builds of ERT that we release on Komodo or PyPI. To
build a release build for development, use `script/build --release`.

### Notes

1. If pip reinstallation fails during the compilation step, try removing the
`_skbuild` directory.

2. The default maximum number of open files is normally relatively low on MacOS
and some Linux distributions. This is likely to make tests crash with mysterious
error-messages. You can inspect the current limits in your shell by issuing he
command `ulimit -a`. In order to increase maximum number of open files, run
`ulimit -n 16384` (or some other large number) and put the command in your
`.profile` to make it persist.

### Testing C code

Install [*ecl*](https://github.com/Equinor/ecl) using CMake as a C library. Then:

``` sh
$ mkdir build
$ cd build
$ cmake ../libres -DBUILD_TESTS=ON
$ cmake --build .
$ ctest --output-on-failure
```

### Building

Use the following commands to start developing from a clean virtualenv
```
$ pip install -r requirements.txt
$ python setup.py develop
```

Alternatively, `pip install -e .` will also setup ERT for development, but
it will be more difficult to recompile the C library.

[scikit-build](https://scikit-build.readthedocs.io/en/latest/index.html) is used
for compiling the C library. It creates a directory named `_skbuild` which is
reused upon future invocations of either `python setup.py develop`, or `python
setup.py build_ext`. The latter only rebuilds the C library. In some cases this
directory must be removed in order for compilation to succeed.

The C library files get installed into `res/.libs`, which is where the
`res` module will look for them.


## Example usage

### Basic ert test
To test if ert itself is working, go to `test-data/local/poly_example` and start ert by running `poly.ert` with `ert gui`
```
cd test-data/local/poly_example
ert gui poly.ert
````
This opens up the ert graphical user interface.
Finally, test ert by starting and successfully running the simulation. 

### ert with a reservoir simulator
To actually get ert to work at your site you need to configure details about
your system; at the very least this means you must configure where your
reservoir simulator is installed. In addition you might want to configure e.g.
queue system in the `site-config` file, but that is not strictly necessary for
a basic test.
