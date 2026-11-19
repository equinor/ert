import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

from ert.__main__ import ert_parser
from ert.cli.main import run_cli as cli_runner
from ert.plugins.plugin_manager import ErtRuntimePlugins, get_site_plugins


def run_cli(*args):
    parser = ArgumentParser(prog="test_main")
    parsed = ert_parser(parser, args)
    return cli_runner(parsed)


def run_cli_with_pm(args: list[Any], runtime_plugins: ErtRuntimePlugins | None = None):
    parser = ArgumentParser(prog="test_main")
    parsed = ert_parser(parser, args)
    if runtime_plugins:
        res = cli_runner(parsed, runtime_plugins)
    else:
        res = cli_runner(parsed, get_site_plugins())
    return res


def read_server_log() -> str:
    """Returns the logs of the ERT servers started by run_cli, oldest first."""
    log_dir = Path(os.environ.get("ERT_LOG_DIR", "."))
    log_files = sorted(log_dir.glob("api-log-storage-*.txt"))
    assert log_files, f"No ERT server log found in {log_dir.absolute()}"
    return "".join(log_file.read_text(encoding="utf-8") for log_file in log_files)
