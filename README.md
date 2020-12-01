# libres [![Libres testing](https://github.com/equinor/libres/workflows/Libres%20testing/badge.svg)](https://github.com/equinor/libres/actions?query=workflow%3A%22Libres+testing%22) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

*libres* is part of the `ERT` project: _[Ensemble based Reservoir Tool](https://github.com/Equinor/ert)_. It is now available in PyPI:

``` sh
$ pip install equinor-libres
```

or, for the latest development version (requires GCC/clang and `Python.h`):

``` sh
$ pip install git+https://github.com/equinor/libres.git@master
```

## Development

*libres* is meant to be installed using `setup.py`, directly or using `pip
install ./`. The `CMakeLists.txt` exists, but is used by `setup.py` to generate
the `libres` C library and by Github Actions to run C tests.

### Building

Use the following commands to start developing from a clean virtualenv
```
$ pip install -r requirements.txt
$ python setup.py develop
```

Alternatively, `pip install -e .` will also setup `libres` for development, but
it will be more difficult to recompile the C library.

[scikit-build](https://scikit-build.readthedocs.io/en/latest/index.html) is used
for compiling the C library. It creates a directory named `_skbuild` which is
reused upon future invocations of either `python setup.py develop`, or `python
setup.py build_ext`. The latter only rebuilds the C library. In some cases this
directory must be removed in order for compilation to succeed.

The C library files get installed into `python/res/.libs`, which is where the
`res` module will look for them.

### Testing Python code

Install the required testing packages and run tests.
```
$ pip install -r test_requirements.txt
$ pytest
```

### Testing C code

Install [*ecl*](https://github.com/Equinor/ecl) using CMake as a C library. Then:

``` sh
$ mkdir build
$ cd build
$ cmake .. -DBUILD_TESTS=ON
$ cmake --build .
$ ctest --output-on-failure
```

## Configuration

### The `site_config` file
As part of the installation process `libres` will install a file called
`site-config` in `share/ert/site-config`; when ert starts this file will be
loaded before the users personal config file. For more extensive use of `ert` it
might be benefical to customize the `site-config` file to your personal site.

To customize, you need to set the environment variable `ERT_SITE_CONFIG` to
point to an alternative file that will be used.

### 6.2 Forward models

`libres` contains basic functionality for forward models to run the reservoir
simulators Eclipse/flow and the geomodelling program RMS. Exactly how these
programs depend on the setup on your site and you must make some modifications
to two files installed with `libres`:

#### 6.2.1. Eclipse/flow configuration

In the Python distribution installed by `libres` there is a file
`res/fm/ecl/ecl_config.yml` which is used to configure the eclipse/flow versions
are available at the location. You can provide an alternative configuration file
by setting the environment variable `ECL_SITE_CONFIG`.

#### 6.2.2. RMS configuration

In the Python distribution installed by `libres` there is a file:
`res/fm/rms/rms_config.yml` which contains some site specific RMS configuration.
You should provide an alternative file with your local path to the `rms` wrapper
script supplied by _Roxar_ by setting the environment variable `RMS_SITE_CONFIG`
to point to the alternative file.
