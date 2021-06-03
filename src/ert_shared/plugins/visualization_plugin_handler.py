class PluginHandlerException(Exception):
    pass


class VisualizationPluginHandler:
    """
    Top-level handler for holding all visualization plugins
    """

    def __init__(self):
        self._plugins = {}

    def add_plugin(self, vis_plugin):
        self._plugins[vis_plugin.name] = vis_plugin

    def get_plugin(self, name):
        try:
            return self._plugins[name]
        except KeyError:
            raise PluginHandlerException(
                f"Visualization plugin: {name} is not installed"
            )
