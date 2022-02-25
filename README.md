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

The following is needed to run ert properly
- libblas
- lapack

MacOS already ships with BLAS and LAPACK implementations in its [vecLib](https://developer.apple.com/documentation/accelerate/veclib) framework.

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
reservoir simulator is installed. This is described in the *Post installation*
section of the [libres README](https://github.com/Equinor/libres). In addition
you might want to configure e.g. queue system in the `site-config` file, but
that is not strictly necessary for a basic test.

In the location `test-data/local/example_case` is a small ert case which can be
used to verify that your installation is basically sound. The example config
file looks like this:
```
-- This ert configuration file is an example which can be used to check that your
-- local ert installation is basically sane. This example is not meant to be used
-- as an automatically run integration test, rather it is meant to be tested
-- interactively. In addition to the compiled application this will also verify that
-- the various configuration files are reasonably correctly stitched together.
--
-- To actually test this invoke the ert binary you have installed and give the path to
-- this file as argument:
--
--    /path/to/installed/ert/bin/ert example.ert
-- 
-- The example is based on the ECLIPSE100 forward model, that implies that you must
-- configure the local eclipse related details corresponding to your site[1].
-- 
-- NB: the current case has *not* been carefully constructed to demonstrate the
-- capabilities of ert; from a model updating perspective the current case is
-- totally uninteresting.
-- 
-- [1]: This amounts to editing the file ecl_config.yml in the res.fm.ecl python
-- package from the libres installation. See the documentation in the
-- ecl_config.yml example file supplied with the libres distribution, or
-- alternatively the "Post install configuration" section in the libres README. 

NUM_REALIZATIONS 20

QUEUE_SYSTEM LOCAL
QUEUE_OPTION LOCAL MAX_RUNNING 4

RUNPATH      output/simulations/runpath/realization-%d/iter-%d
ENSPATH      output/storage

ECLBASE   EXAMPLE%d
DATA_FILE eclipse/model/SPE1.DATA
REFCASE   eclipse/refcase/REFCASE

GEN_KW MULT_PORO templates/poro.tmpl   poro.grdecl  parameters/poro.txt

-- This job will copy the file eclipse/input/schedule to the runpath folder. 
SIMULATION_JOB COPY_FILE eclipse/input/schedule  


-- This forward model job requires that you have eclipse version 2016.02
-- installed locally, feel free to modify this to use a different version if
-- that is what you have installed.
SIMULATION_JOB ECLIPSE100 2016.2 <ECLBASE>

OBS_CONFIG observations/observations.txt


-- This tells ert that you want to load all summary vectors starting with 'W'. 
-- 'F' and 'BPR'. To be able to use the wildcard notation this way you need to 
-- specify a REFCASE.

SUMMARY W*
SUMMARY F*
SUMMARY BPR*
```

**NB: Depending on which reservoir simulator versions you have installed locally
you might have to change the eclipse version number 2016.2 to something else.**

To actually test this go to the `test-data/local/example_case` directory and
start `ert` by giving the full path to the installed binary:

```
   cd test-data/local/example_case
   /local/ert/install/bin/ert example.ert
```

Then the `ert` gui should come up and you can press the `Run simulations`
button. In addition to the gui there is a simple text interface which
can be invoked with the `--text` option.


## Configuration

### The `site_config` file
As part of the installation process ERT will install a file called
`site-config` in `share/ert/site-config`; when ert starts this file will be
loaded before the users personal config file. For more extensive use of `ert` it
might be beneficial to customize the `site-config` file to your personal site.

To customize, you need to set the environment variable `ERT_SITE_CONFIG` to
point to an alternative file that will be used.

### 6.2 Forward models

ERT contains basic functionality for forward models to run the reservoir
simulators Eclipse/flow and the geomodelling program RMS. Exactly how these
programs depend on the setup on your site and you must make some modifications
to two files installed with ERT:

#### 6.2.1. Eclipse/flow configuration

In the Python distribution installed by ERT there is a file
`res/fm/ecl/ecl_config.yml` which is used to configure the eclipse/flow versions
are available at the location. You can provide an alternative configuration file
by setting the environment variable `ECL_SITE_CONFIG`.

#### 6.2.2. RMS configuration

In the Python distribution installed by ERT there is a file:
`res/fm/rms/rms_config.yml` which contains some site specific RMS configuration.
You should provide an alternative file with your local path to the `rms` wrapper
script supplied by _Roxar_ by setting the environment variable `RMS_SITE_CONFIG`
to point to the alternative file.
