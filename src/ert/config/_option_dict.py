import logging
from typing import Dict, Sequence

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

    Errors are reported to the log, and erroring fields ignored:

    >>> import sys
    >>> logger.addHandler(logging.StreamHandler(sys.stdout))
    >>> option_dict(opts + [":T"], 1)
    Ignoring argument :T not properly formatted should be of type ARG:VAL
    {'INPUT_FORMAT': 'ASCII', 'RESULT_FILE': 'file.txt', 'REPORT_STEPS': '3'}

    """
    result = {}
    for option_pair in option_list[offset:]:
        if not isinstance(option_pair, str):
            logger.warning(
                f"Ignoring unsupported option pair{option_pair} "
                f"of type {type(option_pair)}"
            )
            continue

        if len(option_pair.split(":")) == 2:
            key, val = option_pair.split(":")
            if val != "" and key != "":
                result[key] = val
            else:
                logger.warning(
                    f"Ignoring argument {option_pair}"
                    " not properly formatted should be of type ARG:VAL"
                )
    return result
