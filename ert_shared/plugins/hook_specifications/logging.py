from ert_shared.plugins.plugin_manager import hook_specification


@hook_specification
def add_log_handle(logging):
    """
    This hook allows the user to add log hooks and change formatting

    :param logging: A handle to the ert logging instance.
    :type config: :class:`logging.Logger`
    :return: None
    """
