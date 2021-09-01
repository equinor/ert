import subprocess
import shutil
import signal
import sys
import os
import logging
import tempfile
import yaml
import pathlib
from typing import Any, Optional
from webviz_ert.assets import WEBVIZ_CONFIG


logger = logging.getLogger()


def handle_exit(*args: Any) -> None:  # pylint: disable=unused-argument)
    logger.info("\n" + "=" * 32)
    logger.info("Session terminated by the user.\n" "Thank you for using webviz-ert!")
    logger.info("=" * 32)
    sys.tracebacklimit = 0
    sys.stdout = open(os.devnull, "w")
    sys.exit()


def create_config(
    project_identifier: Optional[str], config_file: pathlib.Path, temp_config: Any
) -> None:
    with open(config_file, "r") as f:
        config_dict = yaml.safe_load(f)
        for page in config_dict["pages"]:
            for element in page["content"]:
                for key in element:
                    element[key] = {"project_identifier": project_identifier}
    output_str = yaml.dump(config_dict)
    temp_config.write(str.encode(output_str))
    temp_config.seek(0)


def send_ready():
    """
    Tell ERT's BaseService that we're ready, even though we're not actually
    ready to accept requests. At the moment, ERT doesn't interface with
    webviz-ert in any way, so it's not necessary to send the signal later.
    """
    fd = int(os.environ["ERT_COMM_FD"])
    with os.fdopen(fd, "w") as f:
        f.write("{}")  # Empty, but valid JSON


def run_webviz_ert():
    signal.signal(signal.SIGINT, handle_exit)
    # The entry point of webviz is to call it from command line, and so do we.
    webviz = shutil.which("webviz")
    if webviz:
        send_ready()
        with tempfile.NamedTemporaryFile() as temp_config:
            project_identifier = os.getenv("ERT_PROJECT_IDENTIFIER", os.getcwd())
            if project_identifier is None:
                logger.error("Unable to find ERT project!")
            create_config(project_identifier, WEBVIZ_CONFIG, temp_config)
            os.execl(
                webviz,
                webviz,
                "build",
                temp_config.name,
                "--theme",
                "equinor",
                "--loglevel",
                "DEBUG",
            )
    else:
        logger.error("Failed to find webviz")


if __name__ == "__main__":
    run_webviz_ert()
