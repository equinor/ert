from __future__ import annotations

from typing import TYPE_CHECKING, no_type_check

from ert.plugins.plugin_manager import hook_specification, single_implementation

if TYPE_CHECKING:
    from ert.config import LoginProvider
    from ert.plugins.plugin_response import PluginResponse


@no_type_check
@single_implementation
@hook_specification
def login_provider() -> PluginResponse[type[LoginProvider]]:
    """Provide the login flow that users run with ``ert login``.

    ``ert login`` calls ``login`` on an instance of the returned class. It exits
    with an error if no login provider is installed. If ``login`` raises an
    exception, its message is shown to the user and the traceback is logged.
    Pressing Ctrl-C cancels the login.

    Only one implementation is allowed.

    :return: A subclass of :code:`ert.LoginProvider`.
    """
