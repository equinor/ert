from .plugin_manager import ErtPluginContext, ErtPluginManager
from .visualization_plugin_handler import (
    VisualizationPluginHandler,
    PluginHandlerException,
)


def launch_visualization_plugin(args):
    pm = ErtPluginManager()
    handler = VisualizationPluginHandler()
    pm.hook.register_visualization_plugin(handler=handler)
    try:
        vis_plugin = handler.get_plugin(args.name)
        vis_plugin.run()
    except PluginHandlerException as e:
        raise SystemExit(str(e))
