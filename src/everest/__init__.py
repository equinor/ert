"""
This is the main Everest module.

"""

from .loader import load

try:
    from ert.shared.version import version

    __version__ = version
except ImportError:
    __version__ = "0.0.0"

from everest import detached, docs, jobs, templates, util
from everest.bin.utils import export_to_csv, export_with_progress
from everest.config_keys import ConfigKeys
from everest.export import MetaDataColumnNames, filter_data

__author__ = "Equinor ASA and TNO"
__all__ = [
    "ConfigKeys",
    "MetaDataColumnNames",
    "detached",
    "docs",
    "export_to_csv",
    "export_with_progress",
    "filter_data",
    "jobs",
    "load",
    "templates",
    "util",
]
