import os
import pathlib
from datetime import datetime
from logging import FileHandler
from typing import Any

LOGGING_CONFIG = pathlib.Path(__file__).parent.resolve() / "logger.conf"
STORAGE_LOG_CONFIG = pathlib.Path(__file__).parent.resolve() / "storage_log.conf"


class TimestampedFileHandler(FileHandler):
    def __init__(self, filename: str, *args: Any, **kwargs: Any) -> None:
        timestamp = (
            f"{datetime.now().strftime(datetime.now().strftime('%Y-%m-%dT%H%M'))}"
        )
        filename, extension = os.path.splitext(filename)

        if "ert_config" in kwargs:
            config_file_path = pathlib.Path(kwargs.pop("ert_config"))
            name, ext = os.path.splitext(config_file_path.name)
            config_filename = f"{name}-{ext[1:]}"
            filename = f"{filename}-{config_filename}-{timestamp}{extension}"
        else:
            filename = f"{filename}-{timestamp}{extension}"

        if kwargs.pop("use_log_dir_from_env", False) and "ERT_LOG_DIR" in os.environ:
            log_dir = os.environ["ERT_LOG_DIR"]
            filename = log_dir + "/" + filename

        super().__init__(filename, *args, **kwargs)
