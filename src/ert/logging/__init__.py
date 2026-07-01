import logging
import os
import pathlib
import re
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any

from _ert.utils import file_safe_timestamp

LOGGING_CONFIG = pathlib.Path(__file__).parent.resolve() / "logger.conf"
STORAGE_LOG_CONFIG = pathlib.Path(__file__).parent.resolve() / "storage_log.conf"

_FORMATS_ANSI = {
    logging.FATAL: "\033[31m[FATAL]\033[0m {message}",
    logging.ERROR: "\033[31m[ERROR]\033[0m {message}",
    logging.WARNING: "\033[33m[ WARN]\033[0m {message}",
    logging.INFO: "\033[34m[ INFO]\033[0m {message}",
    logging.DEBUG: "\033[36m[DEBUG]\033[0m {message}",
    logging.NOTSET: "\033[0m        {message}",
}

_FORMATS_NO_COLOR = {
    logging.FATAL: "[FATAL] {message}",
    logging.ERROR: "[ERROR] {message}",
    logging.WARNING: "[ WARN] {message}",
    logging.INFO: "[ INFO] {message}",
    logging.DEBUG: "[DEBUG] {message}",
    logging.NOTSET: "        {message}",
}

_FORMATS = _FORMATS_ANSI if os.isatty(sys.stderr.fileno()) else _FORMATS_NO_COLOR


class TimestampedFileHandler(logging.FileHandler):
    def __init__(self, filename: str, *args: Any, **kwargs: Any) -> None:
        timestamp = file_safe_timestamp(
            datetime.now().astimezone().isoformat(timespec="minutes")
        )

        filename, extension = os.path.splitext(filename)

        if "config_filename" in kwargs:
            config_file_path = pathlib.Path(kwargs.pop("config_filename"))
            name, ext = os.path.splitext(config_file_path.name)
            config_filename = f"{name}-{ext[1:]}"
            filename = f"{filename}-{config_filename}-{timestamp}{extension}"
        else:
            filename = f"{filename}-{timestamp}{extension}"

        if kwargs.pop("use_log_dir_from_env", False) and "ERT_LOG_DIR" in os.environ:
            log_dir = os.environ["ERT_LOG_DIR"]
            filename = log_dir + "/" + filename
            Path(filename).parent.mkdir(exist_ok=True, parents=True)

        super().__init__(filename, *args, **kwargs)


class TerminalFormatter(logging.Formatter):
    """Formats for terminal output

    Specifically, do not output information that is useless or scary for the
    user, like exception tracebacks.

    """

    def __init__(self) -> None:
        super().__init__("%(message)s")

    @staticmethod
    def formatMessage(record: logging.LogRecord) -> str:
        fmt = _FORMATS.get(record.levelno, _FORMATS[logging.NOTSET])
        return fmt.format(message=record.message)

    @staticmethod
    def formatException(
        _: tuple[type[BaseException], BaseException, TracebackType | None]
        | tuple[None, None, None],
    ) -> str:
        return ""


def suppress_logs(logs_to_suppress: list[str]) -> Callable[[logging.LogRecord], bool]:
    """Suppresses logs from loggers listed in logs_to_suppress"""

    def log_filter(record: logging.LogRecord) -> bool:
        for log_name in logs_to_suppress:
            if record.name.startswith(log_name):
                return False
        return True

    return log_filter


class ObjectAttributeFilter(logging.Filter):
    def __init__(self, objects: list[str], attributes: list[str]) -> None:
        super().__init__()
        self._obj_regex = rf"\b(?:{'|'.join(objects)})\([^\)]*\)"
        self._attr_regex = rf"(?:{'|'.join(attributes)})=((?:'[^']*'|[^\s,\)]+))"

    def _redact(self, msg: str | Any) -> str | Any:
        # Record.msg is implemented as str | Any, but should be str in
        # most cases according to the documentation.
        # In our case, it is sometimes an exception, for example TimeoutError.
        # This will return early should the record message not be a str.
        if not isinstance(msg, str):
            return msg

        # Mutating msg will alter the indices of the succeeding- but not
        # preceding text. Reversing the matches allows us to safely mutate
        # the string from back to front.
        for obj_match in reversed(list(re.finditer(self._obj_regex, msg))):
            for attr_match in reversed(
                list(re.finditer(self._attr_regex, obj_match.group()))
            ):
                attr = attr_match.group()
                attr_value = attr.split("=")[1]
                match_censored = re.sub(attr_value, "REDACTED", attr)

                attr_start = obj_match.start() + attr_match.start()
                attr_end = attr_start + len(attr)
                msg = msg[:attr_start] + match_censored + msg[attr_end:]
        return msg

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = self._redact(record.msg)
        return True
