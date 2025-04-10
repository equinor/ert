"""
This is the main Everest module.

"""

try:
    from .loader import load
except Exception as e:
    print(
        f"Error during initialization: {e}\nPlease make sure that "
        "everest is installed correctly and that all dependencies are updated."
    )
    import sys

    sys.exit(1)

try:
    from ert.shared.version import version

    __version__ = version
except ImportError:
    __version__ = "0.0.0"

from everest import detached, jobs, templates, util

__author__ = "Equinor ASA and TNO"
__all__ = [
    "detached",
    "jobs",
    "load",
    "templates",
    "util",
]
