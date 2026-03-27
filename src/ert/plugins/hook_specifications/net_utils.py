from ert.plugins.plugin_manager import hook_specification


@hook_specification
def get_ip_address() -> str:  # type: ignore
    """Ert uses network communication over TCP/IP and needs to provide potential
    network clients with which IP address is to be used to contact the main Ert
    process. By default, Ert will check the operating system routing table to
    get the IP address in use for non-localhost connections.

    On machines exposing several IP addresses, the correct IP is non-trivial
    to pick, and by specifying this hook in an installed plugin, any custom
    code can be injected in order to pick the correct IP."""
