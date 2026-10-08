import argparse
import json
import logging.config
import os
import shutil
import sys
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml

from ert.config import QueueSystem
from ert.logging import LOGGING_CONFIG
from ert.plugins.plugin_manager import ErtPluginManager
from ert.services.ert_client import ErtClient
from ert.storage import (
    ExperimentStatus,
    open_storage,
)
from ert.utils import makedirs_if_needed
from everest.config import EverestConfig
from everest.config.server_config import ServerConfig
from everest.strings import EVEREST


def cleanup_logging() -> None:
    os.environ.pop("ERT_LOG_DIR", None)


@contextmanager
def setup_logging(options: argparse.Namespace) -> Generator[None, None, None]:
    if isinstance(options.config, EverestConfig):
        makedirs_if_needed(Path(options.config.output_dir), roll_if_exists=False)
        log_dir = Path(options.config.output_dir) / "logs"
    else:
        # `everest branch` gives a tuple object here.
        log_dir = Path("logs")

    try:
        log_dir.mkdir(exist_ok=True)
    except PermissionError as err:
        sys.exit(str(err))
    try:
        os.environ["ERT_LOG_DIR"] = str(log_dir)

        config_dict = yaml.safe_load(LOGGING_CONFIG.read_text(encoding="utf-8"))
        if config_dict:
            for handler_name, handler_config in config_dict["handlers"].items():
                if handler_name == "file":
                    handler_config["filename"] = "everest-log.txt"
                if "ert.logging.TimestampedFileHandler" in handler_config.values():
                    handler_config["config_filename"] = ""
                    if isinstance(options.config, EverestConfig):
                        handler_config["config_filename"] = (
                            options.config.config_path.name
                        )
                    else:
                        # `everest branch`
                        handler_config["config_filename"] = options.config[0]
            logging.config.dictConfig(config_dict)

        if "debug" in options and options.debug:
            root_logger = logging.getLogger()
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(logging.DEBUG)
            root_logger.addHandler(handler)

        plugin_manager = ErtPluginManager()
        plugin_manager.add_logging_handle_to_root(logging.getLogger())
        plugin_manager.add_span_processor_to_trace_provider()
        yield
    finally:
        cleanup_logging()


def handle_keyboard_interrupt(signum: int, _: Any, options: argparse.Namespace) -> None:
    width = min(shutil.get_terminal_size(fallback=(78, 24)).columns, 100)
    print("\n" + "=" * width)
    if options.config.server_queue_system == QueueSystem.LOCAL:
        print(
            f"KeyboardInterrupt (ID: {signum}) has been caught. \n"
            "You are running locally. \n"
            "The optimization will be stopped and the program will exit..."
        )
        try:
            client = ErtClient.get_client(
                Path(ServerConfig.get_session_dir(options.config.output_dir)),
                connect_timeout=1,
            )
            if client.server_is_running(timeout=1):
                client.stop_server()
                client.wait_for_server_to_stop(timeout=10)
                print("Server stopped successfully.")

        except TimeoutError:
            print("No running server found.")

    else:
        print(f"KeyboardInterrupt (ID: {signum}) has been caught. Program will exit...")
        config_file = options.config.config_file
        print(
            "You are running in detached mode.\n"
            "To monitor the running optimization use command:\n"
            f"  `everest monitor {config_file}`\n"
            "To kill the running optimization use command:\n"
            f"  `everest kill {config_file}`"
        )
    print("=" * width)
    sys.tracebacklimit = 0
    sys.stdout = open(os.devnull, "w", encoding="utf-8")  # ruff: ignore[builtin-open, open-file-with-context-handler] SIM115
    sys.stderr = open(os.devnull, "w", encoding="utf-8")  # ruff: ignore[builtin-open, open-file-with-context-handler] SIM115
    sys.exit()


def remove_show_scaling_warning_setting() -> None:
    """Remove the now unused "show_scaling_warning" everest preference from
    the legacy ~/.ert preferences file, if present. The whole file is
    deleted if removing it leaves the file empty; otherwise the file is
    rewritten without that key, preserving any other content.
    """
    user_info_path = Path(os.getenv("HOME", "")) / ".ert"
    if not user_info_path.exists():
        return

    logger = logging.getLogger(EVEREST)
    content = user_info_path.read_text(encoding="utf-8")

    try:
        user_info = json.loads(content)
    except json.decoder.JSONDecodeError as e:
        logger.info(
            "Preserving preferences file %s, could not be parsed: %s",
            user_info_path,
            e,
        )
        return

    everest_pref = user_info.get(EVEREST)
    if not isinstance(everest_pref, dict) or "show_scaling_warning" not in everest_pref:
        return

    everest_pref.pop("show_scaling_warning")
    if not everest_pref:
        user_info.pop(EVEREST)

    if not user_info:
        user_info_path.unlink()
        logger.info(
            "Deleted preferences file %s, previously containing: %s",
            user_info_path,
            content,
        )
    else:
        user_info_path.write_text(
            json.dumps(user_info, ensure_ascii=False, indent=4), encoding="utf-8"
        )
        logger.info(
            "Removed legacy show_scaling_warning preference from %s, "
            "previous content: %s, remaining content: %s",
            user_info_path,
            content,
            user_info,
        )


def get_experiment_status(storage_dir: str) -> ExperimentStatus | None:
    """
    Reads the experiment status from storage. We assume that there is
    only one experiment for each everest run in storage.
    """
    with open_storage(storage_dir, "r") as storage:
        experiments = list(storage.experiments)
        return None if not experiments else experiments[0].status


class ArgParseFormatter(argparse.HelpFormatter):
    def _fill_text(self, text: str, width: int, indent: str) -> str:
        return "\n\n".join(
            [
                argparse.HelpFormatter._fill_text(self, p, width, indent)
                for p in text.split("\n\n")
            ]
        )
