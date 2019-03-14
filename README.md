# ert [![Build Status](https://travis-ci.org/equinor/ert.svg?branch=master)](https://travis-ci.org/equinor/ert)

ERT - Ensemble based Reservoir Tool - is a tool to run ensemble based on
reservoir models. ERT was originally devised as tool to do model updating
(history matching) with the EnKF method, now the primary method for model
updating is the Ensemble Smoother (ES).


The `ert` program is based on three different repositories:

1. [libecl](https://github.com/Equinor/libecl) which contains utilities to read and write Eclipse files.

2. [libres](https://github.com/Equinor/libres) utilities to manage reservoir data, and algorithms do actually do model updating.

3. ert - this repository - the actual application and all of the GUI.

##  Building ert

#### 1. Build and install [libecl](https://github.com/Equinor/libecl) and [libres](https://github.com/Equinor/libres). 
When configuring `libecl` and
`libres` you should used the option `-DCMAKE_INSTALL_PREFIX` to tell ``cmake``
where to install. The value passed to `CMAKE_INSTALL_PREFIX` will be needed when
running cmake to in point 4 below. For now let us assume that the prefix
`/local/ert/install` was used.


#### 2. Install Python dependencies

```
pip install -r requirements.txt 
```

In addition you will need to install `PyQt4` - this package can not be installed
using `pip`, you should probably use the package manager from your operating
system.

#### 3. Update environment variables 
To ensure that the build system correctly finds the `ecl` Python package you
need to set the environment variables `PYTHONPATH` and `LD_LIBRARY_PATH` to
include the `libecl` installation:
  
```
bash% export LD_LIBRARY_PATH=/local/ert/install/lib64:$LD_LIBRARY_PATH
bash% export PYTHONPATH=/local/ert/install/lib/python2.7/site-packages:$PYTHONPATH
```

Observe that path components `lib64` and `lib/python2.7/site-packages` will
depend on your Python version and which Linux distribution you are using. The
example given here is for RedHat based distributions.


#### 4. Run `cmake` to configure `ert`

When running `cmake` you must tell `cmake` where the `libecl` code is located with
`-DCMAKE_PREFIX_PATH`  i.e. in addition to other possible
arguments you must at least add:

```
-DCMAKE_PREFIX_PATH=/local/ert/install
-DCMAKE_INSTALL_PREFIX=/local/ert/install
```

in addition you probably want to pass `-DCMAKE_INSTALL_PREFIX` to configure where
the `ert` distribuion should be installed, normally that will be the same
location where you have already installed `libecl` and `libres`. 


#### 5. Run `make` to compile `ert`

After you have run cmake you should run `make` and `make install` to build and install `ert`:

```
bash% make
bash% make install
```

When this process if over you will have a binary executable `ert` installed in
`/local/ert/install/bin/ert`. 


#### 6. Try your new `ert` installation

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
