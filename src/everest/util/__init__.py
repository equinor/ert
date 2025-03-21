import logging
import os
from datetime import UTC, datetime

from ropt.version import version as ropt_version

try:
    from ert.shared.version import version as ert_version
except ImportError:
    ert_version = "0.0.0"
from everest.strings import DATE_FORMAT, EVEREST


def version_info() -> str:
    return f"everest:{ert_version}, ropt:{ropt_version}, ert:{ert_version}"


def date2str(date: datetime) -> str:
    return datetime.strftime(date, DATE_FORMAT)


def str2date(date_str: str) -> datetime:
    return datetime.strptime(date_str, DATE_FORMAT)


def makedirs_if_needed(path: str, roll_if_exists: bool = False) -> None:
    if os.path.isdir(path):
        if not roll_if_exists:
            return
        _roll_dir(path)  # exists and should be rolled
    os.makedirs(path)


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


def _roll_dir(old_name: str) -> None:
    old_name = os.path.realpath(old_name)
    new_name = old_name + datetime.now(UTC).strftime("__%Y-%m-%d_%H.%M.%S.%f")
    os.rename(old_name, new_name)
    logging.getLogger(EVEREST).info(f"renamed {old_name} to {new_name}")
