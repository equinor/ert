"""
This is the main Everest module.

"""

from .loader import load

try:
    from .version import version

    __version__ = version
except ImportError:
    __version__ = "0.0.0"

from everest import detached, docs, jobs, templates, util
from everest.bin.utils import export_to_csv, export_with_progress
from everest.config_keys import ConfigKeys
from everest.export import MetaDataColumnNames, export, filter_data, validate_export
from everest.suite import (  # flake8: noqa F401
    SIMULATOR_END,
    SIMULATOR_START,
    SIMULATOR_UPDATE,
    start_optimization,
)

__author__ = "Equinor ASA and TNO"
__all__ = [
    "load",
    "start_optimization",
    "SIMULATOR_START",
    "SIMULATOR_UPDATE",
    "SIMULATOR_END",
    "ConfigKeys",
    "export",
    "export_with_progress",
    "export_to_csv",
    "validate_export",
    "filter_data",
    "MetaDataColumnNames",
    "detached",
    "docs",
    "jobs",
    "templates",
    "util",
]
