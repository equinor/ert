from __future__ import annotations

from typing import TYPE_CHECKING

from ert.plugins.plugin_manager import hook_specification, single_implementation

if TYPE_CHECKING:
    from ert.run_models.event import StatusEvents


@single_implementation
@hook_specification
def on_status_event(event: StatusEvents, experiment_id: str) -> None:
    """Called for every status event emitted by a running experiment.

    Implementations are called synchronously from the server's event loop,
    so they must return quickly.

    Only one implementation is allowed.

    :param event: The emitted status event.
    :param experiment_id: The server's ID of the experiment run that emitted
        the event. This is not the ID of the experiment in storage.
    """
