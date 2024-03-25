from unittest.mock import MagicMock

import pytest

from ert.config import CancelPluginException, ErtPlugin


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
        def run(self, ert_script):
            return ert_script

    plugin = FixturePlugin()
    fixture_mock = MagicMock()
    assert plugin.initializeAndRun([], [], {"ert_script": fixture_mock}) == fixture_mock


def test_plugin_with_fixtures_and_arguments():
    class FixturePlugin(ErtPlugin):
        def run(self, arg_1, ert_script, arg_2, fixture_2):
            return arg_1, ert_script, arg_2, fixture_2

    plugin = FixturePlugin()
    fixture_mock = MagicMock()
    fixture2_mock = MagicMock()
    assert plugin.initializeAndRun(
        [], [1, 2], {"ert_script": fixture_mock, "fixture_2": fixture2_mock}
    ) == ("1", fixture_mock, "2", fixture2_mock)
