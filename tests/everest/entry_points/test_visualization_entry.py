from unittest.mock import patch

import pluggy

from everest.bin.visualization_script import visualization_entry
from everest.config import EverestConfig
from everest.detached import ServerStatus
from everest.plugins import hook_impl, hook_specs
from everest.strings import EVEREST
from tests.everest.utils import capture_streams


class MockPluginManager(pluggy.PluginManager):
    def __init__(self):
        super().__init__(EVEREST)
        self.add_hookspecs(hook_specs)
        self.register(hook_impl)


@patch(
    "everest.bin.visualization_script.everserver_status",
    return_value={"status": ServerStatus.completed},
)
@patch("everest.bin.visualization_script.EverestPluginManager", MockPluginManager)
def test_expected_message_when_no_visualisation_plugin_is_installed(
    _, change_to_tmpdir, min_config
):
    config_file = "test.yml"
    config = EverestConfig(**min_config)
    config.dump(config_file)
    with capture_streams() as (out, _):
        visualization_entry([config_file])
    assert "No visualization plugin installed!" in out.getvalue()
