from argparse import ArgumentParser

from ert.__main__ import ert_parser
from ert.cli.main import run_cli as cli_runner
from ert.shared.feature_toggling import FeatureToggling


def run_cli(*args):
    parser = ArgumentParser(prog="test_main")
    parsed = ert_parser(parser, args)
    FeatureToggling.update_from_args(parsed)
    res = cli_runner(parsed)
    FeatureToggling.reset()
    return res
