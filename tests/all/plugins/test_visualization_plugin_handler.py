import dummy_plugins
import pytest

from ert_shared.plugins import PluginHandlerException, VisualizationPluginHandler
from ert_shared.plugins.plugin_manager import ErtPluginManager


def test_visualization_plugin_handler():
    pm = ErtPluginManager(plugins=[dummy_plugins])
    handler = VisualizationPluginHandler()
    pm.hook.register_visualization_plugin(handler=handler)
    assert "example" in handler._plugins
    handler.get_plugin("example")
    with pytest.raises(PluginHandlerException):
        handler.get_plugin("not_existing")
