import os
import tempfile
from unittest.mock import Mock

import pytest

from ert.shared.plugins import ErtPluginContext
from ert.shared.plugins.plugin_manager import _ErtPluginManager
from tests.unit_tests.all.plugins import dummy_plugins

env_vars = [
    "ECL100_SITE_CONFIG",
    "ECL300_SITE_CONFIG",
    "FLOW_SITE_CONFIG",
    "RMS_SITE_CONFIG",
    "ERT_SITE_CONFIG",
]


def plugin_context(plugins):
    ctx = ErtPluginContext()
    ctx.plugin_manager = _ErtPluginManager(plugins)
    return ctx


def test_no_plugins(monkeypatch):
    # pylint: disable=pointless-statement
    monkeypatch.delenv("ERT_SITE_CONFIG", raising=False)
    with plugin_context(plugins=[]) as c:
        with pytest.raises(KeyError):
            os.environ["ECL100_SITE_CONFIG"]
        with pytest.raises(KeyError):
            os.environ["ECL300_SITE_CONFIG"]
        with pytest.raises(KeyError):
            os.environ["FLOW_SITE_CONFIG"]
        with pytest.raises(KeyError):
            os.environ["RMS_SITE_CONFIG"]

        assert os.path.isfile(os.environ["ERT_SITE_CONFIG"])
        with open(os.environ["ERT_SITE_CONFIG"], encoding="utf-8") as f:
            assert c.plugin_manager.get_site_config_content() == f.read()

        path = os.environ["ERT_SITE_CONFIG"]

    with pytest.raises(KeyError):
        os.environ["ERT_SITE_CONFIG"]
    assert not os.path.isfile(path)


def test_with_plugins(monkeypatch):
    # pylint: disable=pointless-statement
    monkeypatch.delenv("ERT_SITE_CONFIG", raising=False)
    # We are comparing two function calls, both of which generate a tmpdir,
    # this makes sure that the same tmpdir is called on both occasions.
    monkeypatch.setattr(tempfile, "mkdtemp", Mock(return_value=tempfile.mkdtemp()))
    with plugin_context(plugins=[dummy_plugins]) as c:
        with pytest.raises(KeyError):
            os.environ["ECL100_SITE_CONFIG"]
        with pytest.raises(KeyError):
            os.environ["ECL300_SITE_CONFIG"]
        with pytest.raises(KeyError):
            os.environ["FLOW_SITE_CONFIG"]
        with pytest.raises(KeyError):
            os.environ["RMS_SITE_CONFIG"]

        assert os.path.isfile(os.environ["ERT_SITE_CONFIG"])
        with open(os.environ["ERT_SITE_CONFIG"], encoding="utf-8") as f:
            assert c.plugin_manager.get_site_config_content() == f.read()

        path = os.environ["ERT_SITE_CONFIG"]

    with pytest.raises(KeyError):
        os.environ["ERT_SITE_CONFIG"]
    assert not os.path.isfile(path)


def test_already_set(monkeypatch):
    for var in env_vars:
        monkeypatch.setenv(var, "TEST")

    with plugin_context(plugins=[dummy_plugins]):
        for var in env_vars:
            assert os.environ[var] == "TEST"

    for var in env_vars:
        assert os.environ[var] == "TEST"
