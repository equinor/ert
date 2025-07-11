from pathlib import Path
from unittest.mock import patch

import pluggy
import pytest

from everest.bin.visualization_script import visualization_entry
from everest.config import EverestConfig
from everest.plugins import hook_impl, hook_specs
from everest.strings import EVEREST
from tests.everest.utils import capture_streams


class MockPluginManager(pluggy.PluginManager):
    def __init__(self) -> None:
        super().__init__(EVEREST)
        self.add_hookspecs(hook_specs)
        self.register(hook_impl)


@pytest.mark.xdist_group("math_func/config_advanced.yml")
@patch("everest.bin.visualization_script.EverestPluginManager", MockPluginManager)
def test_expected_message_when_no_visualisation_plugin_is_installed(
    change_to_tmpdir, cached_example, min_config
):
    config_path, config_file, _, _ = cached_example("math_func/config_advanced.yml")
    config = EverestConfig.load_file(Path(config_path) / config_file)

    config_file = "test.yml"
    config = EverestConfig(**min_config)
    config.dump(config_file)
    with capture_streams() as (out, _):
        visualization_entry([config_file])
    assert "No visualization plugin installed!" in out.getvalue()
