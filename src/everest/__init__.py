"""
This is the main Everest module.

"""

try:
    from .loader import load
except Exception as e:
    print(
        f"Error during initialization: {e}\nPlease make sure that everest is installed correctly and that all dependencies are updated."
    )
    import sys

    sys.exit(1)

try:
    from ert.shared.version import version

    __version__ = version
except ImportError:
    __version__ = "0.0.0"

from everest import detached, jobs, templates, util
from everest.bin.utils import export_to_csv, export_with_progress
from everest.export import MetaDataColumnNames, filter_data

__author__ = "Equinor ASA and TNO"
__all__ = [
    "MetaDataColumnNames",
    "detached",
    "export_to_csv",
    "export_with_progress",
    "filter_data",
    "jobs",
    "load",
    "templates",
    "util",
]
