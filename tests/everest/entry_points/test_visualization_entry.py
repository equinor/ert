from unittest.mock import patch

import pluggy
import pytest

from everest.bin.visualization_script import visualization_entry
from everest.plugins import hook_impl, hook_specs
from everest.strings import EVEREST
from tests.everest.utils import (
    capture_streams,
    skipif_no_everviz,
)


class MockPluginManager(pluggy.PluginManager):
    def __init__(self) -> None:
        super().__init__(EVEREST)
        self.add_hookspecs(hook_specs)
        self.register(hook_impl)


@pytest.mark.slow
@pytest.mark.xdist_group("math_func/config_advanced.yml")
@patch("everest.bin.visualization_script.EverestPluginManager", MockPluginManager)
def test_expected_message_when_no_visualisation_plugin_is_installed(cached_example):
    _, config_file, _, _ = cached_example("math_func/config_advanced.yml")
    with capture_streams() as (out, _):
        visualization_entry([config_file])
    assert "No visualization plugin installed!" in out.getvalue()


@skipif_no_everviz
@pytest.mark.slow
@pytest.mark.xdist_group("math_func/config_minimal.yml")
@patch("subprocess.call")
def test_that_everviz_can_setup_everviz_config(mock_call, cached_example):
    _, config_file, _, _ = cached_example("math_func/config_minimal.yml")
    with capture_streams() as (out, _):
        visualization_entry([config_file])
    assert "Default everviz config created:" in out.getvalue()
