from __future__ import annotations

from typing import TYPE_CHECKING

from ert.plugins.plugin_manager import hook_specification, single_implementation

if TYPE_CHECKING:
    from ert.run_models.event import StatusEvents


@single_implementation
@hook_specification
def on_status_event(event: StatusEvents, experiment_id: str) -> None:
    """Called for every status event emitted by a running experiment.

    Implementations are called from a dedicated thread, one event at a time
    and in the order the events were emitted. A slow implementation does not
    delay the server, but it does delay its own subsequent events.

    Only one implementation is allowed.

    :param event: The emitted status event.
    :param experiment_id: The server's ID of the experiment run that emitted
        the event. This is not the ID of the experiment in storage.
    """
