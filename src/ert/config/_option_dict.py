import logging
from typing import Dict, Sequence

from .parsing import ConfigValidationError

logger = logging.getLogger(__name__)


def option_dict(option_list: Sequence[str], offset: int) -> Dict[str, str]:
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
