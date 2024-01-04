# ert

[![Build Status](https://github.com/equinor/ert/actions/workflows/build.yml/badge.svg)](https://github.com/equinor/ert/actions/workflows/build.yml)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ert)](https://img.shields.io/pypi/pyversions/ert)
[![Code Style](https://github.com/equinor/ert/actions/workflows/style.yml/badge.svg)](https://github.com/equinor/ert/actions/workflows/style.yml)
[![Type checking](https://github.com/equinor/ert/actions/workflows/typing.yml/badge.svg)](https://github.com/equinor/ert/actions/workflows/typing.yml)
[![codecov](https://codecov.io/gh/equinor/ert/graph/badge.svg?token=keVAcWavZ1)](https://codecov.io/gh/equinor/ert)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

ert - Ensemble based Reservoir Tool - is designed for running
ensembles of dynamical models such as reservoir models,
in order to do sensitivity analysis and data assimilation.
ert supports data assimilation using the Ensemble Smoother (ES),
Ensemble Smoother with Multiple Data Assimilation (ES-MDA) and
Iterative Ensemble Smoother (IES).

## Installation

``` sh
$ pip install ert
$ ert --help
```

or, for the latest development version (requires Python development headers):

``` sh
$ pip install git+https://github.com/equinor/ert.git@main
$ ert --help
```

For examples and help with configuration, see the [ert Documentation](https://ert.readthedocs.io/en/latest/getting_started/configuration/poly_new/guide.html#configuration-guide).

### Installing on Macs with ARM CPUs

A few of ert's dependencies aren't compiled for ARM CPUs. Because of this,
we need to do some Rosetta "hot swapping".

First, install Rosetta by running `softwareupdate --install-rosetta [--agree-to-license]`

Once Rosetta is installed, you can switch to an Intel based architecture by running:
`arch -x86_64 <SHELL_PATH>`. Note that if your shell is installed
as an ARM executable, this will error. If that's the case, you can simply pass
`/bin/zsh` as the shell path.

Now you're set to install Homebrew for Intel architectures:

`/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`

Now, to be able to hot swap between Intel and ARM architectures, add the following
to your shell profile config:

```sh
alias arm="env /usr/bin/arch -arm64 <SHELL_PATH> --login"
alias intel="env /usr/bin/arch -x86_64 <SHELL_PATH> --login"

local cpu=$(uname -m)

if [[ $cpu == "arm64" ]]; then
	eval "$(/opt/homebrew/bin/brew shellenv)"
fi

if [[ $cpu == "x86_64" ]]; then
	eval "$(/usr/local/homebrew/bin/brew shellenv)"
fi
```

Note: You can always check which architecture you're running by calling either
`arch` or `uname -m`.

This will allow you to switch between architectures by calling either `intel` or `arm`
from your terminal. Switching architectures will automatically source the correct
Hombrew executable for your architecture as well, which is key.

Now, simply switch to Intel, and install Python and set up a virtualenv as
instructed below.

## Developing

ert was originally written in C/C++ but most new code is Python.

### Developing Python

You might first want to make sure that some system level packages are installed
before attempting setup:

```
- pip
- python include headers
- (python) venv
- (python) setuptools
- (python) wheel
```

It is left as an exercise to the reader to figure out how to install these on
their respective system.

To start developing the Python code, we suggest installing ert in editable mode
into a [virtual environment](https://docs.python.org/3/library/venv.html) to
isolate the install (substitute the appropriate way of sourcing venv for your shell):

```sh
# Create and enable a virtualenv
python3 -m venv my_virtualenv
source my_virtualenv/bin/activate

# Update build dependencies
pip install --upgrade pip wheel setuptools

# Download and install ert
git clone https://github.com/equinor/ert
cd ert
pip install --editable .
```

### Test setup

Additional development packages must be installed to run the test suite:

```sh
pip install "ert[dev]"
pytest tests/
```

[Git LFS](https://git-lfs.com/) must be installed to get all the files. This is packaged as `git-lfs` on Ubuntu, Fedora or macOS Homebrew. For Equinor RGS node users, it is possible to use `git` from Red Hat Software Collections:
```sh
source /opt/rh/rh-git227/enable
```
test-data/block_storage is a submodule and must be checked out.
```sh
git submodule update --init --recursive
```

If you checked out submodules without having git lfs installed, you can force git lfs to run in all submodules with:
```sh
git submodule foreach "git lfs pull"
```


### Style requirements

There are a set of style requirements, which are gathered in the `pre-commit`
configuration, to have it automatically run on each commit do:

``` sh
$ pip install pre-commit
$ pre-commit install
```

### Trouble with setup

If you encounter problems during install, try deleting the `_skbuild` folder before reinstalling.

As a simple test of your `ert` installation, you may try to run one of the
examples, for instance:

```
cd test-data/poly_example
# for non-gui trial run
ert test_run poly.ert
# for gui trial run
ert gui poly.ert
```

Note that in order to parse floating point numbers from text files correctly,
your locale must be set such that `.` is the decimal separator, e.g. by setting

```
# export LC_NUMERIC=en_US.UTF-8
```

in bash (or an equivalent way of setting that environment variable for your
shell).

### Developing C++

C++ is the backbone of ert as in used extensively in important parts of ert.
There's a combination of legacy code and newer refactored code. The end goal is
likely that some core performance-critical functionality will be implemented in
C++ and the rest of the business logic will be implemented in Python.

While running `--editable` will create the necessary Python extension module
(`src/ert/_clib.cpython-*.so`), changing C++ code will not take effect even when
reloading ert. This requires recompilation, which means reinstalling ert from
scratch.

To avoid recompiling already-compiled source files, we provide the
`script/build` script. From a fresh virtualenv:

```sh
git clone https://github.com/equinor/ert
cd ert
script/build
```

This command will update `pip` if necessary, install the build dependencies,
compile ert and install in editable mode, and finally install the runtime
requirements. Further invocations will only build the necessary source files. To
do a full rebuild, delete the `_skbuild` directory.

Note: This will create a debug build, which is faster to compile and comes with
debugging functionality enabled. This means that, for example, Eigen
computations will be checked and will abort if preconditions aren't met (eg.
when inverting a matrix, it will explicitly check that the matrix is square).
The downside is that this makes the code unoptimised and slow. Debugging flags
are therefore not present in builds of ert that we release on Komodo or PyPI. To
build a release build for development, use `script/build --release`.

### Notes

1. If pip reinstallation fails during the compilation step, try removing the
`_skbuild` directory.

2. The default maximum number of open files is normally relatively low on MacOS
and some Linux distributions. This is likely to make tests crash with mysterious
error-messages. You can inspect the current limits in your shell by issuing the
command `ulimit -a`. In order to increase maximum number of open files, run
`ulimit -n 16384` (or some other large number) and put the command in your
`.profile` to make it persist.

### Running C++ tests

The C++ code and tests require [resdata](https://github.com/Equinor/resdata). As long
as you have `pip install resdata`'d into your Python virtualenv all should work.

``` sh
# Create and enable a virtualenv
python3 -m venv my_virtualenv
source my_virtualenv/bin/activate

# Install build dependencies
pip install pybind11 conan cmake resdata

# Build ert and tests
mkdir build && cd build
cmake ../src/clib -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)

# Run tests
ctest --output-on-failure
```

## Example usage

### Basic ert test
To test if ert itself is working, go to `test-data/poly_example` and start ert by running `poly.ert` with `ert gui`
```
cd test-data/poly_example
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
