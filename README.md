# libres [![Build Status](https://travis-ci.org/Equinor/libres.svg?branch=master)](https://travis-ci.org/Equinor/libres)

`libres` is part of the `ERT` project: _[Ensemble based Reservoir Tool](https://github.com/Equinor/ert)_.

## Building libres

### 1. Build libecl
Build and install [libecl](https://github.com/Equinor/libecl). When configuring
`libecl` you should used the option `-DCMAKE_INSTALL_PREFIX` to tell ``cmake``
where to install `libecl`. The value passed to `CMAKE_INSTALL_PREFIX` will be
needed when running cmake to configure `libres` in point 4 below. For now let us
assume that the prefix `/local/ert/install` was used.
   
   
### 2. Install Python dependencies

```
pip install -r requirements.txt 
```

### 3. Update environment variables 
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


### 4. Run `cmake` to configure `libres`

When running `cmake` you must tell `cmake` where the `libecl` code is located
with `-DCMAKE_PREFIX_PATH`, i.e. in addition to other possible arguments you
must at least add:

```
-DCMAKE_PREFIX_PATH=/local/ert/install
```

in addition you probably want to pass `-DCMAKE_INSTALL_PREFIX` to configure where
the `libres` distribuion should be installed, normally that will be the same
location where you have already installed `libecl`. 


### 5. Run `make` to compile `libres`

After you have run cmake you should run `make` and `make install` to build and install `libres`:

```
bash% make
bash% make install
```

### 6. postinstall configuration


#### 6.1. The `site_config` file
As part of the installation process `libres` will install a file called
`site-config` in `$prefix/share/ert/site-config`; when ert starts this file will
be loaded before the users personal config file. For more extensive use of `ert`
it might be benefical to customize the `site-config` file to your personal site.
There are three possible ways to do this:

1. You can just edit the installed file manually - `libres` will not install
   it's version of the `site-config` file if one is already present.
   
2. The path to `site-config` file is *compiled into* the `libres` library, if
   you pass the `cmake` option `-DSITE_CONFIG_FILE=/path/to/site/config` when
   configuring `cmake`; that way your own personal `site-config` file is built
   in.
   
3. If you set the environment variable `ERT_SITE_CONFIG` to point to an
   alternative file that will be used when bootstrapping. This can be a handy
   way to debug the `site-config` settings.
   
For a start you can probably just use the shipped default version of the
`site-config` file.


#### 6.2 Forward models

The `libres` code contains basic functionality for forward models to run the
reservoir simulators Eclipse/flow and the geomodelling program RMS. Exactly how
these programs depend on the setup on your site and you must make some
modifications to two files installed with `libres`:

##### 6.2.1. Eclipse/flow configuration

In the Python distribution installed by `libres` there is a file:
`res/fm/ecl/ecl_config.yml` which is used to configure the eclipse/flow versions
are available at the location. You should edit this file to correspond to the
conditions at your site; alternatively you can store an alternative
configuration file elsewhere and set the environment variable `ECL_SITE_CONFIG`
to point to the alternative file.


##### 6.2.2. RMS configuration

In the Python distribution installed by `libres` there is a file:
`res/fm/rms/rms_config.yml` which contains some site specific RMS configuration.
You should update this file with your local path to the `rms` wrapper script
supplied by `Roxar`; alternatively you can store an alternative configuration
file elseswhere and set the environment variable `RMS_SITE_CONFIG` to point to
the alternative file.
