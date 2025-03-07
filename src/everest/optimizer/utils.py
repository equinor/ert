import importlib

from ropt.plugins import PluginManager


def get_ropt_plugin_manager() -> PluginManager:
    # To check the validity of optimization and sampler backends and their
    # supported algorithms or methods, an instance of a ropt PluginManager is
    # needed. Everest also needs a ropt plugin manager at runtime which may add
    # additional optimization and/or sampler backends. To be sure that these
    # added backends are detected, all code should use this function to access
    # the plugin manager. Any optimizer/sampler plugins that need to be added at
    # runtime should be added in this function.
    #
    # Note: backends can also be added via the Python entrypoints mechanism,
    # these are detected by default and do not need to be added here.

    try:
        return PluginManager()
    except Exception as exc:
        ert_version = importlib.metadata.version("ert")
        ropt_version = importlib.metadata.version("ropt")
        msg = (
            f"Error while initializing ropt:\n\n{exc}.\n\n"
            "Check the everest installation, there may a be version mismatch.\n"
            f"  (ERT: {ert_version}, ropt: {ropt_version})\n"
            "If the everest installation is correct, please report this as a bug."
        )
        raise RuntimeError(msg) from exc
