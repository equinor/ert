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

*ERT* is Python and C software. To start developing, install it in editable
mode:

```
$ git clone https://github.com/equinor/ert
$ cd ert
$ pip install -e .
```

Additional development packages must be installed to run the test suite:
```
$ pip install -r dev-requirements.txt
$ pytest tests/
```

<strong>For Mac-users</strong> <em>The default maximum number of open files is normally relatively low on MacOS.
This is likely to make tests crash with mysterious error-messages.
You can inspect the current limits in your shell by issuing he command 'ulimit -a'.
In order to increase maximum number of open files, run 'ulimit -n 16384' (or some other large number)
and put the command in your .profile to make it persist.
</em>

ERT is meant to be installed using `setup.py`, directly or using `pip
install ./`. The `CMakeLists.txt` in libres exists, but is used by `setup.py`
to generate the ERT C library (the C library formerly known as *libres*) and
by Github Actions to run C tests.

ERT requires a recent version of `pip` - hence you are advised to upgrade
your `pip` installation with

```sh
$ pip install --upgrade pip
```
If your `pip` version is too old the installation of ERT will fail, and the error messages will be incomprehensible.

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
