import pathlib
import os
from logging import FileHandler
from datetime import datetime

LOGGING_CONFIG = pathlib.Path(__file__).parent.resolve() / "logger.conf"
STORAGE_LOG_CONFIG = pathlib.Path(__file__).parent.resolve() / "storage_log.conf"


class TimestampedFileHandler(FileHandler):
    def __init__(self, filename: str, *args, **kwargs) -> None:
        filename, extension = os.path.splitext(filename)
        filename = (
            f"{filename}-"
            f"{datetime.now().strftime(datetime.now().strftime('%Y-%m-%dT%H%M'))}"
            f"{extension}"
        )
        super().__init__(filename, *args, **kwargs)
