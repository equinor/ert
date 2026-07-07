import re
import sys

BLACK = "\033[30m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
CYAN = "\033[36m"
RESET = "\033[0m"

CURSOR_UP = "\033[1A"
CLEAR_LINE = "\033[2K"

ALL_CODES = [
    BLACK,
    RED,
    GREEN,
    YELLOW,
    BLUE,
    CYAN,
    RESET,
    CURSOR_UP,
    CLEAR_LINE,
]

_ESCAPE_RE = re.compile(r"\033\[[0-?]*[ -/]*[@-~]")


def ansi_print(msg: str) -> None:
    if not sys.stdout.isatty():
        msg = _ESCAPE_RE.sub("", msg)
    print(msg)
