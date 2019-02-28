#  Copyright (C) 2011  Equinor ASA, Norway.
#
#  The file '__init__.py' is part of ERT - Ensemble based Reservoir Tool.
#
#  ERT is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  ERT is distributed in the hope that it will be useful, but WITHOUT ANY
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or
#  FITNESS FOR A PARTICULAR PURPOSE.
#
#  See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html>
#  for more details.
"""
Ert - Ensemble Reservoir Tool - a package for reservoir modeling.

The res package itself has no code, but contains several subpackages:

res.job_queue:

The res package is based on wrapping the libriaries from the ERT C
code with ctypes; an essential part of ctypes approach is to load the
shared libraries with the ctypes.CDLL() function. The ctypes.CDLL()
function uses the standard methods of the operating system,
i.e. standard locations configured with ld.so.conf and the environment
variable LD_LIBRARY_PATH.

To avoid conflict with other application using the res libraries the
Python code should be able to locate the shared libraries without
(necessarily) using the LD_LIBRARY_PATH variable. The default
behaviour is to try to load from the library ../../lib64, but by using
the enviornment variable ERT_LIBRARY_PATH you can alter how ert looks
for shared libraries.

   1. By default the code will try to load the shared libraries from
      '../../lib64' relative to the location of this file.

   2. Depending on the value of ERT_LIBRARY_PATH two different
      behaviours can be imposed:

         Existing path: the package will look in the path pointed to
            by ERT_LIBRARY_PATH for shared libraries.

         Arbitrary value: the package will use standard load order for
         the operating system.

If the fixed path, given by the default ../../lib64 or ERT_LIBRARY_PATH
alternative fails, the loader will try the default load behaviour
before giving up completely.
"""
import os.path
import sys

import warnings
warnings.simplefilter('always', DeprecationWarning) # see #1437

from cwrap import load as cwrapload
from cwrap import Prototype

try:
    import ert_site_init
except ImportError:
    pass


required_version_hex = 0x02070000

res_lib_path = None
ert_so_version = ""
__version__ = "0.0.0"


# 1. Try to load the __res_lib_info module; this module has been
#    configured by cmake during the build configuration process. The
#    module should contain the variable lib_path pointing to the
#    directory with shared object files.
try:
    from .__res_lib_info import ResLibInfo
    res_lib_path = ResLibInfo.lib_path
    ert_so_version = ResLibInfo.so_version
    __version__ = ResLibInfo.__version__
except ImportError:
    pass
except AttributeError:
    pass


# 2. Using the environment variable ERT_LIBRARY_PATH it is possible to
#    override the default algorithms. If the ERT_LIBRARY_PATH is set
#    to a non existing directory a warning will go to stderr and the
#    setting will be ignored.
env_lib_path = os.getenv("ERT_LIBRARY_PATH")
if env_lib_path:
    if os.path.isdir( env_lib_path ):
        res_lib_path = os.getenv("ERT_LIBRARY_PATH")
    else:
        sys.stderr.write("Warning: Environment variable ERT_LIBRARY_PATH points to nonexisting directory:%s - ignored" % env_lib_path)


# Check that the final ert_lib_path setting corresponds to an existing
# directory.
if res_lib_path:
    if not os.path.isabs(res_lib_path):
        res_lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), res_lib_path))

    if not os.path.isdir( res_lib_path ):
        res_lib_path = None


if sys.hexversion < required_version_hex:
    raise Exception("ERT Python requires Python 2.7")

# This load() function is *the* function actually loading shared
# libraries.

def load(name):
    return cwrapload(name, path=res_lib_path, so_version=ert_so_version)

class ResPrototype(Prototype):
    lib = load("libres")

    def __init__(self, prototype, bind=True):
        super(ResPrototype, self).__init__(ResPrototype.lib, prototype, bind=bind)

RES_LIB = ResPrototype.lib

from res.util import ResVersion
from ecl.util.util import updateAbortSignals

updateAbortSignals( )

def root():
    """
    Will print the filesystem root of the current ert package.
    """
    return os.path.abspath( os.path.join( os.path.dirname( __file__ ) , "../"))
