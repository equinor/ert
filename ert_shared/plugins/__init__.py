import os
import sys
from .plugin_manager import ErtPluginContext, ErtPluginManager
from .visualization_plugin_handler import (
    VisualizationPluginHandler,
    PluginHandlerException,
)
from ert_shared.storage.connection import autostart


def launch_visualization_plugin(args):
    try:
        os.environ["ERT_PROJECT_IDENTIFIER"] = autostart(args.project)
    except RuntimeError:
        sys.exit("Failed to connect to ERT Storage server")

    pm = ErtPluginManager()
    handler = VisualizationPluginHandler()
    pm.hook.register_visualization_plugin(handler=handler)
    try:
        vis_plugin = handler.get_plugin(args.name)
        vis_plugin.run()
    except PluginHandlerException as e:
        raise SystemExit(str(e))
