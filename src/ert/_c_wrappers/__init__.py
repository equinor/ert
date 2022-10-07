"""
Ert - Ensemble Reservoir Tool - a package for reservoir modeling.
"""
import os.path
import warnings

import ecl  # This needs to be here for ... reasons

warnings.filterwarnings(action="always", category=DeprecationWarning, module=r"res|ert")

from cwrap import Prototype  # noqa

try:
    from ._version import version as __version__
except ImportError:
    pass


def _load_lib():
    import ctypes

    import ert._clib

    lib = ctypes.CDLL(ert._clib.__file__)  # pylint: disable=no-member

    return lib


class ResPrototype(Prototype):
    lib = _load_lib()

    def __init__(self, prototype, bind=True):
        super().__init__(ResPrototype.lib, prototype, bind=bind)


RES_LIB = ResPrototype.lib

from ecl.util.util import updateAbortSignals  # noqa

updateAbortSignals()


def root():
    """
    Will print the filesystem root of the current ert package.
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
