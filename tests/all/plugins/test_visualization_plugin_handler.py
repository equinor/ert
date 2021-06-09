import pytest
from ert_shared.plugins.plugin_manager import ErtPluginManager
from ert_shared.plugins import VisualizationPluginHandler, PluginHandlerException
from . import dummy_plugins


def test_visualization_plugin_handler():
    pm = ErtPluginManager(plugins=[dummy_plugins])
    handler = VisualizationPluginHandler()
    pm.hook.register_visualization_plugin(handler=handler)
    assert "example" in handler._plugins
    handler.get_plugin("example")
    with pytest.raises(PluginHandlerException):
        handler.get_plugin("not_existing")
