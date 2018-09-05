# libres [![Build Status](https://travis-ci.org/Statoil/libres.svg?branch=master)](https://travis-ci.org/Statoil/libres)

`libres` is part of the `ERT` project: _[Ensemble based Reservoir Tool](https://github.com/Statoil/ert)_.

## Building libres

#### 1. Build libecl
Build and install [libecl](https://github.com/Statoil/libecl). When configuring
`libecl` you should used the option `-DCMAKE_INSTALL_PREFIX` to tell ``cmake``
where to install `libecl`. The value passed to `CMAKE_INSTALL_PREFIX` will be
needed when running cmake to configure `libres` in point 4 below. For now let us
assume that the prefix `/local/ert/install` was used.
   
   
#### 2. Install Python dependencies

```
pip install -r requirements.txt 
```

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


#### 4. Run `cmake` to configure `libres`

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


#### 5. Run `make` to compile `libres`

After you have run cmake you should run `make` and `make install` to build and install `libres`:

```
bash% make
bash% make install
```
