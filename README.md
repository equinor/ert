# ert [![Build Status](https://travis-ci.org/Statoil/ert.svg?branch=master)](https://travis-ci.org/Statoil/ert)

ERT - Ensemble based Reservoir Tool - is a tool to run ensemble based on
reservoir models. ERT was originally devised as tool to do model updating
(history matching) with the EnKF method, now the primary method for model
updating is the Ensemble Smoother (ES).


The `ert` program is based on three different repositories:

1. [libecl](https://github.com/Statoil/libecl) which contains utilities to read and write Eclipse files.

2. [libres](https://github.com/Statoil/libres) utilities to manage reservoir data, and algorithms do actually do model updating.

3. ert - this repository - the actual application and all of the GUI.

##  Building ert

#### 1. Build and install [libecl](https://github.com/Statoil/libecl) and [libres](https://github.com/Statoil/libres). 
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
`-DCMAKE_PREFIX_PATH` and also where some `Python/cmake` interoperability
modules are found with `-DCMAKE_MODULE_PATH`, i.e. in addition to other possible
arguments you must at least add:

```
-DCMAKE_PREFIX_PATH=/local/ert/install \
-DCMAKE_MODULE_PATH=/local/ert/install/share/cmake/Modules \
```

in addition you probably want to pass `-DCMAKE_INSTALL_PREFIX` to configure where
the `libres` distribuion should be installed, normally that will be the same
location where you have already installed `libecl`. 


#### 5. Run `make` to compile `ert`

After you have run cmake you should run `make` and `make install` to build and install `ert`:

```
bash% make
bash% make install
```

When this process if over you will have a binary executable `ert` installed in
`/local/ert/install/bin/ert`. To try this out go to the root our `ert/` source
folder and type:

```
/local/ert/install/bin/ert test-data/local/snake_oil/snake_oil.ert
```

where of course the `/local/ert/install` part should be replaced with the
location where you have actually installed `ert`.
