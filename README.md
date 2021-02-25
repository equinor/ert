# ert

![Build status](https://github.com/equinor/ert/actions/workflows/build.yml/badge.svg)
![Code style](https://github.com/equinor/ert/actions/workflows/style.yml/badge.svg)
![Type Hinting](https://github.com/equinor/ert/actions/workflows/style.yml/typing.svg)
![ERT2 test data setups](https://github.com/equinor/ert/actions/workflows/run_ert2_test_data_setups.yml/badge.svg)
![ERT3 polynomial demo](https://github.com/equinor/ert/actions/workflows/run_polynomial_demo.yml/badge.svg)
![ERT3 SPE1 demo](https://github.com/equinor/ert/actions/workflows/run_spe1_demo.yml/badge.svg)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

ERT - Ensemble based Reservoir Tool - is a tool to run ensemble based on
reservoir models. ERT was originally devised as tool to do model updating
(history matching) with the EnKF method, now the primary method for model
updating is the Ensemble Smoother (ES).


``` sh
$ pip install ert
$ ert --help
```

or, for the latest development version:

``` sh
$ pip install git+https://github.com/equinor/ert.git@master
$ ert --help
```


The `ert` program is based on three different repositories:

1. [ecl](https://github.com/Equinor/ecl) which contains utilities to read and write Eclipse files.

2. [libres](https://github.com/Equinor/libres) utilities to manage reservoir data, and algorithms do actually do model updating.

3. ert - this repository - the actual application and all of the GUI.


ERT is now Python 3 only. The last Python 2 compatible release is [2.14](https://github.com/equinor/ert/tree/version-2.14)

## Developing

ERT is pure Python software. To start developing, install it in editable mode:

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

## Example usage

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

RUNPATH      output/simulations/runpath/realisation-%d/iter-%d
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
