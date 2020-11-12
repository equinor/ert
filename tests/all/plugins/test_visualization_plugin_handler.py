import pytest
from ert_shared.plugins.plugin_manager import (
    ErtPluginContext,
    ErtPluginManager,
    hook_implementation,
)
from ert_shared.plugins.plugin_response import plugin_response
from ert_shared.plugins import VisualizationPluginHandler, PluginHandlerException
import tests.all.plugins.dummy_plugins as dummy_plugins


def test_visualization_plugin_handler():
    pm = ErtPluginManager(plugins=[dummy_plugins])
    handler = VisualizationPluginHandler()
    pm.hook.register_visualization_plugin(handler=handler)
    assert "example" in handler._plugins
    vis_plugin = handler.get_plugin("example")
    with pytest.raises(PluginHandlerException):
        handler.get_plugin("not_existing")
