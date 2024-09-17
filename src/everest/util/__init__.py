import datetime
import logging
import os

from ropt.version import version as ropt_version

from ert.shared.version import version as ert_version
from everest.plugins.hook_manager import EverestPluginManager
from everest.strings import DATE_FORMAT, DEFAULT_LOGGING_FORMAT
from everest.util.async_run import async_run  # noqa

try:
    import opm.io
    from opm.io.ecl_state import EclipseState
    from opm.io.schedule import Schedule

    def has_opm():
        return True

except ImportError:

    def has_opm():
        return False


def get_azure_logging_handler():
    pm = EverestPluginManager()
    handles = pm.hook.add_log_handle_to_root()
    if handles:
        return handles[0]


def configure_logger(
    name=None,
    file_path=None,
    log_level=None,
    formatter=DEFAULT_LOGGING_FORMAT,
    log_to_azure=False,
) -> logging.Logger:
    logger = logging.getLogger(name)

    logger.setLevel(log_level or logging.INFO)
    if file_path is not None:
        makedirs_if_needed(os.path.dirname(file_path))
        handler = logging.FileHandler(file_path)
        handler.setFormatter(logging.Formatter(formatter))
        logger.addHandler(handler)

    # Setup azure logging if needed
    azure_handler = get_azure_logging_handler()
    if log_to_azure and azure_handler:
        logger.addHandler(azure_handler)

    return logger


def version_info():
    return ("everest:'{}'\nropt:'{}'\nert:'{}'").format(
        ert_version, ropt_version, ert_version
    )


def date2str(date):
    return datetime.datetime.strftime(date, DATE_FORMAT)


def str2date(date_str):
    return datetime.datetime.strptime(date_str, DATE_FORMAT)


def makedirs_if_needed(path, roll_if_exists=False):
    if os.path.isdir(path):
        if not roll_if_exists:
            return
        _roll_dir(path)  # exists and should be rolled
    os.makedirs(path)


def _roll_dir(old_name):
    old_name = os.path.realpath(old_name)
    new_name = old_name + datetime.datetime.utcnow().strftime("__%Y-%m-%d_%H.%M.%S.%f")
    os.rename(old_name, new_name)
    logging.getLogger("everest").info("renamed %s to %s" % (old_name, new_name))


def load_deck(fname):
    """Take a .DATA file and return an opm.io.Deck."""
    if not os.path.exists(fname):
        raise IOError('No such data file "%s".' % fname)

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


def read_wellnames(fname):
    """Take a .DATA file and return the list of well
    names at time the first timestep from deck."""
    deck = load_deck(fname)
    state = EclipseState(deck)
    schedule = Schedule(deck, state)
    return [str(well.name) for well in schedule.get_wells(0)]


def read_groupnames(fname):
    """Take a .DATA file and return the list of group
    names at the first timestep from deck."""
    deck = load_deck(fname)
    state = EclipseState(deck)
    schedule = Schedule(deck, state)
    return [str(group.name) for group in schedule._groups(0)]
