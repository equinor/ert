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
import ecl  # This needs to be here for ... reasons

import warnings

warnings.filterwarnings(action="always", category=DeprecationWarning, module=r"res|ert")

try:
    from ._version import version as __version__
except ImportError:
    pass


from res._lib.exports import ResPrototype

def _setup_site_config():
    from res._lib import set_site_config
    from pathlib import Path

    path = Path(__file__).parent
    for p in path.parents:
        npath = p / "ert_shared" / "share" / "ert" / "site-config"
        if npath.is_file():
            path = npath
            break
    else:
        raise ImportError("Could not find `share/ert/site-config`")
    set_site_config(str(path))

_setup_site_config()


from ecl.util.util import updateAbortSignals

updateAbortSignals()


def root():
    """
    Will print the filesystem root of the current ert package.
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))


__all__ = ["ResPrototype"]
