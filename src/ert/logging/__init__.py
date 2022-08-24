import os
import pathlib
from datetime import datetime
from logging import FileHandler

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

        if kwargs.pop("use_log_dir_from_env", False) and "ERT_LOG_DIR" in os.environ:
            log_dir = os.environ["ERT_LOG_DIR"]
            filename = log_dir + "/" + filename

        super().__init__(filename, *args, **kwargs)
