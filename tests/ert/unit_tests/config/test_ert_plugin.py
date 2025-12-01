import logging
from unittest.mock import MagicMock

import pytest

from ert.plugins import CancelPluginException, ErtPlugin


class SimplePlugin(ErtPlugin):
    def run(self, parameter1, parameter2):
        assert parameter1 == "one"
        assert parameter2 == 2

    def getArguments(self, parent=None):
        return ["one", 2]


class FullPlugin(ErtPlugin):
    def getName(self):
        return "FullPlugin"

    def getDescription(self):
        return "Fully described!"

    def run(self, arg1, arg2, arg3=None):
        assert arg1 == 5
        assert arg2 == "No"
        assert arg3 is None

    def getArguments(self, parent=None):
        return [5, "No"]


class CanceledPlugin(ErtPlugin):
    def run(self, arg1):
        pass

    def getArguments(self, parent=None):
        raise CancelPluginException("Cancel test!")


def test_simple_ert_plugin():
    simple_plugin = SimplePlugin()

    arguments = simple_plugin.getArguments()

    assert "SimplePlugin" in simple_plugin.getName()
    assert simple_plugin.getDescription() == "No description provided!"

    simple_plugin.initializeAndRun([str, int], arguments)


def test_full_ert_plugin():
    plugin = FullPlugin()

    assert plugin.getName() == "FullPlugin"
    assert plugin.getDescription() == "Fully described!"

    arguments = plugin.getArguments()

    plugin.initializeAndRun([int, str, float], arguments)


def test_cancel_plugin():
    plugin = CanceledPlugin()

    with pytest.raises(CancelPluginException):
        plugin.getArguments()


def test_plugin_with_fixtures():
    class FixturePlugin(ErtPlugin):
        def run(self, ensemble):
            return ensemble

    plugin = FixturePlugin()
    fixture_mock = MagicMock()
    assert plugin.initializeAndRun([], [], {"ensemble": fixture_mock}) == fixture_mock


def test_plugin_with_missing_arguments(caplog):
    class FixturePlugin(ErtPlugin):
        def run(self, arg_1, ensemble, run_paths, arg_2="something"):
            pass

    plugin = FixturePlugin()
    fixture_mock = MagicMock()
    fixture2_mock = MagicMock()
    with caplog.at_level(logging.WARNING):
        plugin.initializeAndRun(
            [],
            [1, 2],
            {"ensemble": fixture_mock, "run_paths": fixture2_mock},
        )

    assert plugin.hasFailed()
    log = "\n".join(caplog.messages)
    assert "FixturePlugin misconfigured" in log
    assert ("arguments: ['arg_1', 'arg_2'] not found in fixtures") in log


def test_plugin_with_fixtures_and_enough_arguments():
    class FixturePlugin(ErtPlugin):
        def run(self, workflow_args, ensemble):
            return workflow_args, ensemble

    plugin = FixturePlugin()
    fixture_mock = MagicMock()
    assert plugin.initializeAndRun([], [1, 2, 3], {"ensemble": fixture_mock}) == (
        ["1", "2", "3"],
        fixture_mock,
    )


def test_plugin_with_default_arguments():
    class FixturePlugin(ErtPlugin):
        def run(self, ensemble=None):
            return ensemble

    plugin = FixturePlugin()
    fixture_mock = MagicMock()
    assert plugin.initializeAndRun([], [], {"ensemble": fixture_mock}) == fixture_mock


def test_plugin_with_args():
    class FixturePlugin(ErtPlugin):
        def run(self, *args):
            return args

    plugin = FixturePlugin()
    fixture_mock = MagicMock()
    assert plugin.initializeAndRun([], [1, 2], {"ensemble": fixture_mock}) == (
        "1",
        "2",
    )


def test_plugin_with_args_and_kwargs():
    class FixturePlugin(ErtPlugin):
        def run(self, *args, **kwargs):
            return args

    plugin = FixturePlugin()
    fixture_mock = MagicMock()
    assert plugin.initializeAndRun([], [1, 2], {"ensemble": fixture_mock}) == (
        "1",
        "2",
    )
