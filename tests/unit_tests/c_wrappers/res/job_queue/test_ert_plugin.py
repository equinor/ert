import pytest

from ert.job_queue import CancelPluginException, ErtPlugin


class SimplePlugin(ErtPlugin):
    def run(self, parameter1, parameter2):  # pylint: disable=arguments-differ
        assert parameter1 == "one"
        assert parameter2 == 2

    def getArguments(self, parent=None):
        return ["one", 2]


class FullPlugin(ErtPlugin):
    def getName(self):
        return "FullPlugin"

    def getDescription(self):
        return "Fully described!"

    def run(self, arg1, arg2, arg3=None):  # pylint: disable=arguments-differ
        assert arg1 == 5
        assert arg2 == "No"
        assert arg3 is None

    def getArguments(self, parent=None):
        return [5, "No"]


class CanceledPlugin(ErtPlugin):
    def run(self, arg1):  # pylint: disable=arguments-differ
        pass

    def getArguments(self, parent=None):
        raise CancelPluginException("Cancel test!")


def test_simple_ert_plugin():
    simple_plugin = SimplePlugin("ert", storage=None)

    arguments = simple_plugin.getArguments()

    assert "SimplePlugin" in simple_plugin.getName()
    assert simple_plugin.getDescription() == "No description provided!"

    simple_plugin.initializeAndRun([str, int], arguments)


def test_full_ert_plugin():
    plugin = FullPlugin("ert", storage=None)

    assert plugin.getName() == "FullPlugin"
    assert plugin.getDescription() == "Fully described!"

    arguments = plugin.getArguments()

    plugin.initializeAndRun([int, str, float], arguments)


def test_cancel_plugin():
    plugin = CanceledPlugin("ert", storage=None)

    with pytest.raises(CancelPluginException):
        plugin.getArguments()
