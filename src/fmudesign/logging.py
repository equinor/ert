import argparse
import logging.config
import os
import sys
from collections.abc import Generator
from contextlib import contextmanager
from logging import Logger
from pathlib import Path

import yaml

from ert.logging import LOGGING_CONFIG
from ert.plugins import ErtPluginManager


def log_and_print(msg: str, logger: Logger, level: int = logging.INFO) -> None:
    logger.log(level, msg)
    print(msg)


@contextmanager
def setup_logging(options: argparse.Namespace) -> Generator[None, None, None]:
    log_dir = Path("logs")

    try:
        os.environ["ERT_LOG_DIR"] = str(log_dir)
        config_dict = yaml.safe_load(LOGGING_CONFIG.read_text(encoding="utf-8"))
        if config_dict:
            for handler_name, handler_config in config_dict["handlers"].items():
                if handler_name == "file":
                    handler_config["filename"] = "fmudesign-log.txt"
                if (
                    "ert.logging.TimestampedFileHandler" in handler_config.values()
                    and (
                        config := getattr(options, "config", None)
                        or getattr(options, "file", None)
                    )
                    is not None
                ):
                    handler_config["config_filename"] = str(config).replace("_", "-")
            try:
                logging.config.dictConfig(config_dict)
            except ValueError as err:
                if "handler 'file'" in str(err):
                    exit_msg = (
                        f"Could not configure log handler for files. "
                        f"Check if you have write-access to the logs-directory "
                        f"({Path(log_dir).resolve()})."
                    )
                else:
                    exit_msg = str(err)
                os.environ.pop("ERT_LOG_DIR")
                sys.exit(exit_msg)

        plugin_manager = ErtPluginManager()
        plugin_manager.add_logging_handle_to_root(logging.getLogger())
        plugin_manager.add_span_processor_to_trace_provider()
        yield
    finally:
        os.environ.pop("ERT_LOG_DIR", None)
