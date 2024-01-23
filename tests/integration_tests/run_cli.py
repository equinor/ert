from argparse import ArgumentParser

from ert.__main__ import ert_parser
from ert.cli.main import run_cli as cli_runner


def run_cli(*args):
    parser = ArgumentParser(prog="test_main")
    parsed = ert_parser(parser, args)
    return cli_runner(parsed)
