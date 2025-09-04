from argparse import ArgumentParser
from typing import Any

from ert.__main__ import ert_parser
from ert.cli.main import run_cli as cli_runner
from ert.plugins.plugin_manager import ErtPluginContext, ErtRuntimePlugins

RUNTIME_PLUGINS = ErtPluginContext().__enter__()  # noqa: PLC2801


def run_cli(*args):
    parser = ArgumentParser(prog="test_main")
    parsed = ert_parser(parser, args)
    res = cli_runner(parsed)
    return res


def run_cli_with_pm(args: list[Any], runtime_plugins: ErtRuntimePlugins | None = None):
    parser = ArgumentParser(prog="test_main")
    parsed = ert_parser(parser, args)
    res = cli_runner(parsed, runtime_plugins or RUNTIME_PLUGINS)
    return res
