from ert.plugins.plugin_manager import hook_specification


@hook_specification
def activate_script() -> str:  # type: ignore
    """
    Allows the plugin to provide a script that will be run when
    the driver submits to the cluster. The script will run in
    bash.

    Example:
    import ert

    @ert.plugin(name="my_plugin")
    def activate_script():
        return "source /private/venv/my_env/bin/activate

    :return: Activate script
    """
