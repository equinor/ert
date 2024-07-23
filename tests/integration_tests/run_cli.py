from argparse import ArgumentParser
from typing import Any, List

import pytest

from ert.__main__ import ert_parser
from ert.cli.main import run_cli as cli_runner
from ert.plugins import ErtPluginManager
from ert.shared.feature_toggling import FeatureScheduler


def run_cli(*args):
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(FeatureScheduler, "_value", None)
        parser = ArgumentParser(prog="test_main")
        parsed = ert_parser(parser, args)
        FeatureScheduler.set_value(parsed)
        res = cli_runner(parsed)
        return res


def run_cli_with_pm(args: List[Any], pm: ErtPluginManager):
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(FeatureScheduler, "_value", None)
        parser = ArgumentParser(prog="test_main")
        parsed = ert_parser(parser, args)
        FeatureScheduler.set_value(parsed)
        res = cli_runner(parsed, pm)
        return res
