import logging

from .parsing import ConfigValidationError

logger = logging.getLogger(__name__)


def str_to_bool(txt: str) -> bool:
    """This function converts text to boolean values according to the rules of
    the FORWARD_INIT keyword.

    The rules for str_to_bool is keep for backwards compatability

    First, any upper/lower case true/false value is converted to the corresponding
    boolean value:

    >>> str_to_bool("TRUE")
    True
    >>> str_to_bool("true")
    True
    >>> str_to_bool("True")
    True
    >>> str_to_bool("FALSE")
    False
    >>> str_to_bool("false")
    False
    >>> str_to_bool("False")
    False
    """
    if txt.lower() == "true":
        return True
    elif txt.lower() == "false":
        return False
    else:
        raise ConfigValidationError.with_context(f"Invalid boolean value {txt!r}", txt)
