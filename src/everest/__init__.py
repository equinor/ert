"""
This is the main Everest module.

"""

try:
    from ert.shared.version import version

    __version__ = version
except ImportError:
    __version__ = "0.0.0"

from everest import detached, templates, util

__author__ = "Equinor ASA and TNO"
__all__ = [
    "detached",
    "templates",
    "util",
]
