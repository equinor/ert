import os
from .plugin_manager import ErtPluginContext, ErtPluginManager
from .visualization_plugin_handler import (
    VisualizationPluginHandler,
    PluginHandlerException,
)
from ert_shared.storage.connection import get_info, get_project_id
from ert_shared.storage.server_monitor import ServerMonitor


def launch_visualization_plugin(args):

    # Try to use project specified in args (Defaults to cwd)
    try:
        get_info(args.project)
        os.environ["ERT_PROJECT_IDENTIFIER"] = os.path.realpath(args.project)
    except RuntimeError as e:
        # Try to use "registered" ert server
        try:
            path = get_project_id()
            os.environ["ERT_PROJECT_IDENTIFIER"] = str(path.absolute())
        except RuntimeError as e:
            # try to start ert api
            monitor = ServerMonitor.get_instance()
            monitor.start()
            monitor.fetch_connection_info()
            os.environ["ERT_PROJECT_IDENTIFIER"] = os.path.realpath(os.getcwd())

    pm = ErtPluginManager()
    handler = VisualizationPluginHandler()
    pm.hook.register_visualization_plugin(handler=handler)
    try:
        vis_plugin = handler.get_plugin(args.name)
        vis_plugin.run()
    except PluginHandlerException as e:
        raise SystemExit(str(e))
