import logging

from ropt.version import version as ropt_version

try:
    from ert.shared.version import version as ert_version
except ImportError:
    ert_version = "0.0.0"

from everest.strings import EVEREST


def version_info() -> str:
    return f"everest:{ert_version}, ropt:{ropt_version}, ert:{ert_version}"


def warn_user_that_runpath_is_nonempty() -> None:
    print(
        "Everest is running in an existing runpath.\n\n"
        "Please be aware of the following:\n"
        "- Previously generated results "
        "might be overwritten.\n"
        "- Previously generated files might "
        "be used if not configured correctly.\n"
    )
    logging.getLogger(EVEREST).warning("Everest is running in an existing runpath")
