from __future__ import annotations

from typing import TYPE_CHECKING, no_type_check

from ert.plugins.plugin_manager import hook_specification, single_implementation

if TYPE_CHECKING:
    from ert.config import AuthProvider
    from ert.plugins.plugin_response import PluginResponse


@no_type_check
@single_implementation
@hook_specification
def auth_provider() -> PluginResponse[type[AuthProvider]]:
    """Provide the login and logout flows that users run with ``ert auth``.

    ``ert auth login`` and ``ert auth logout`` call ``login`` and ``logout``
    on an instance of the returned class. They exit with an error if no auth
    provider is installed. If ``login`` or ``logout`` raises an exception, its
    message is shown to the user and the traceback is logged. Pressing Ctrl-C
    cancels the command.

    The auth provider only handles login and logout. Ert does not
    pass it to other plugins, so plugins that call authenticated services
    must get their tokens from their own authentication library, for
    example by reading the token cache written during login.

    Only one implementation is allowed.

    :return: A subclass of :code:`ert.AuthProvider`.
    """
