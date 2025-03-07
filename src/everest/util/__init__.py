import logging
import os
from datetime import UTC, datetime

from ropt.version import version as ropt_version

try:
    from ert.shared.version import version as ert_version
except ImportError:
    ert_version = "0.0.0"
from everest.strings import DATE_FORMAT, EVEREST

try:
    import opm.io
    from opm.io.ecl_state import EclipseState
    from opm.io.schedule import Schedule

    def has_opm() -> bool:
        return True

except ImportError:

    def has_opm() -> bool:
        return False


def version_info() -> str:
    return f"everest:'{ert_version}'\nropt:'{ropt_version}'\nert:'{ert_version}'"


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


def load_deck(fname: str):  # type: ignore
    """Take a .DATA file and return an opm.io.Deck."""
    if not os.path.exists(fname):
        raise OSError(f'No such data file "{fname}".')

    if not has_opm():
        raise RuntimeError("Cannot load ECL files, opm could not be imported")

    # OPM parser will fail with different errors on corrupted Eclipse input files.
    # We should ignore these as we just want to extract the wells, we don't
    # care about anything else in the file (for now).
    errors_to_ignore = [
        "INTERNAL_ERROR_UNINITIALIZED_THPRES",
        "PARSE_EXTRA_DATA",
        "PARSE_MISSING_DIMS_KEYWORD",
        "PARSE_MISSING_INCLUDE",
        "PARSE_RANDOM_SLASH",
        "PARSE_RANDOM_TEXT",
        "PARSE_UNKNOWN_KEYWORD",
        "SUMMARY_UNKNOWN_GROUP",
        "SUMMARY_UNKNOWN_WELL",
        "UNSUPPORTED_COMPORD_TYPE",
        "UNSUPPORTED_INITIAL_THPRES",
        "UNSUPPORTED_SCHEDULE_GEO_MODIFIER",
        "UNSUPPORTED_TERMINATE_IF_BHP",
    ]
    parse_context = opm.io.ParseContext(
        [(err_name, opm.io.action.ignore) for err_name in errors_to_ignore]
    )
    return opm.io.Parser().parse(fname, parse_context)


def read_wellnames(fname: str) -> list[str]:
    """Take a .DATA file and return the list of well
    names at time the first timestep from deck."""
    deck = load_deck(fname)
    state = EclipseState(deck)
    schedule = Schedule(deck, state)
    return [str(well.name) for well in schedule.get_wells(0)]


def read_groupnames(fname: str) -> list[str]:
    """Take a .DATA file and return the list of group
    names at the first timestep from deck."""
    deck = load_deck(fname)
    state = EclipseState(deck)
    schedule = Schedule(deck, state)
    return [str(group.name) for group in schedule._groups(0)]
