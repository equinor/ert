import logging
from collections.abc import Sequence

from .config_errors import ConfigValidationError

logger = logging.getLogger(__name__)


def parse_variable_options(
    line: list[str], max_positionals: int
) -> tuple[list[str], dict[str, str]]:
    """
    This function is responsible for taking a config line and splitting it
    into positional arguments and named arguments in cases were the number
    of positional arguments vary.
    """
    offset = next(
        (
            i
            for i, val in enumerate(line)
            if isinstance(val, str) and len(val.split(":")) == 2
        ),
        max_positionals,
    )
    kwargs = option_dict(line, offset)
    args = line[:offset]
    return args, kwargs


def option_dict(option_list: Sequence[str], offset: int) -> dict[str, str]:
    """Gets the list of options given to a keywords such as GEN_DATA.

    The first step of parsing will separate a line such as

      GEN_DATA NAME INPUT_FORMAT:ASCII RESULT_FILE:file.txt REPORT_STEPS:3

    into

    >>> opts = ["NAME", "INPUT_FORMAT:ASCII", "RESULT_FILE:file.txt", "REPORT_STEPS:3"]

    From there, option_dict can be used to get a dictionary of the options:

    >>> option_dict(opts, 1)
    {'INPUT_FORMAT': 'ASCII', 'RESULT_FILE': 'file.txt', 'REPORT_STEPS': '3'}
    """
    result = {}
    for option_pair in option_list[offset:]:
        if len(option_pair.split(":")) == 2:
            key, val = option_pair.split(":")
            if val and key:
                result[key] = val
            else:
                raise ConfigValidationError.with_context(
                    f"Invalid argument {option_pair!r}", option_pair
                )
        else:
            raise ConfigValidationError.with_context(
                f"Invalid argument {option_pair!r}", option_pair
            )
    return result
