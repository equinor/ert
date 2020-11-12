from ert_shared.plugins.plugin_manager import hook_specification


@hook_specification
def register_visualization_plugin(handler):
    """
    This hook allow the user to register a visualization plugin for ert.
    The response expect to add a plugin using handler.add_plugin(<class>).
    The class expects to have a static property name and and static method run.

    :param handler: A handler for the installed plugins.
    :type handler: :class: `ert_shared.plugins.visualization_plugin_handler.VisualizationPluginHandler`
    :return: None
    """
