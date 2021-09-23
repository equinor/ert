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
"""
import os.path
import sys
import platform
import ecl

import warnings

warnings.filterwarnings(action="always", category=DeprecationWarning, module=r"res|ert")

from cwrap import Prototype

try:
    from ._version import version as __version__
except ImportError:
    pass


def _load_lib():
    import ctypes

    # Find and dlopen libres
    lib_path = os.path.join(os.path.dirname(__file__), ".libs")
    if not os.path.isdir(lib_path):
        lib_path = ""
    lib_path = os.path.join(lib_path, "libres.so")

    lib = ctypes.CDLL(lib_path, ctypes.RTLD_GLOBAL)

    # Configure site_config to be a ctypes.CFUNCTION with type:
    # void set_site_config(char *);
    site_config = lib.set_site_config
    site_config.restype = None
    site_config.argtypes = (ctypes.c_char_p,)

    # Find share/ert
    from pathlib import Path

    path = Path(__file__).parent
    for p in path.parents:
        npath = p / "share" / "ert" / "site-config"
        if npath.is_file():
            path = npath
            break
    else:
        raise ImportError("Could not find `share/ert/site-config`")

    # Set site-config to point to [PREFIX]/share/ert/site-config
    site_config(str(path).encode("utf-8"))

    # Configure set_analysis_modules_dir to be a ctypes.CFUNCTION with type:
    # void set_analysis_modules_dir(char *);
    set_analysis_modules_dir = lib.set_analysis_modules_dir
    set_analysis_modules_dir.restype = None
    set_analysis_modules_dir.argtypes = (ctypes.c_char_p,)

    # Set analysis modules dir to be [CURRENT DIR]/.libs
    path = os.path.join(os.path.dirname(__file__), ".libs")
    set_analysis_modules_dir(path.encode("utf-8"))

    return lib


class ResPrototype(Prototype):
    lib = _load_lib()

    def __init__(self, prototype, bind=True):
        super(ResPrototype, self).__init__(ResPrototype.lib, prototype, bind=bind)


RES_LIB = ResPrototype.lib

from res.util import ResVersion
from ecl.util.util import updateAbortSignals

updateAbortSignals()


def root():
    """
    Will print the filesystem root of the current ert package.
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
