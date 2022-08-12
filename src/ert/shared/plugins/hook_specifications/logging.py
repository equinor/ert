from ert.shared.plugins.plugin_manager import hook_specification


@hook_specification
def add_log_handle_to_root():
    """
    Create a log handle which will be added to the root logger
    in the main entry point.

    :return: A log handle that will be added to the root logger
    :rtype: logging.Handler
    """
