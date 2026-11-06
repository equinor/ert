from __future__ import annotations

from typing import TYPE_CHECKING, no_type_check

from ert.plugins.plugin_manager import hook_specification

if TYPE_CHECKING:
    from ert.run_models.event import StatusEvents


@no_type_check
@hook_specification
def on_status_event(event: StatusEvents) -> None:
    """Called for every status event emitted by a running experiment.

    Implementations are called synchronously by the experiment, so they must
    return quickly. Events may be emitted concurrently from multiple threads,
    so implementations must be thread-safe.
    The execution of a plugin does not influence any others;
    if an exception is raised the error will be logged for that plugin, and the
    rest will continue as expected.

    :param event: The emitted status event.
    """
