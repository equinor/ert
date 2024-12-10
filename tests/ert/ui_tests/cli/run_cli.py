from argparse import ArgumentParser
from typing import Any

from ert.__main__ import ert_parser
from ert.cli.main import run_cli as cli_runner
from ert.plugins import ErtPluginManager


def run_cli(*args):
    parser = ArgumentParser(prog="test_main")
    parsed = ert_parser(parser, args)
    res = cli_runner(parsed)
    return res


def run_cli_with_pm(args: list[Any], pm: ErtPluginManager | None = None):
    if pm is None:
        pm = ErtPluginManager()
    parser = ArgumentParser(prog="test_main")
    parsed = ert_parser(parser, args)
    res = cli_runner(parsed, pm)
    return res
