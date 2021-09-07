from ert_shared.plugins.plugin_manager import hook_specification


@hook_specification
def add_log_handle(logging):
    """
    Modify ert's logging instance, for example adding handlers and changing formatting.

    :param logging: A handle to the ert logging instance.
    :type logging: :module:`logging`
    :return: None
    """
