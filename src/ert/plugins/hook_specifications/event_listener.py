from __future__ import annotations

from typing import TYPE_CHECKING

from ert.plugins.plugin_manager import hook_specification, single_implementation

if TYPE_CHECKING:
    from ert.run_models.event import StatusEvents


@single_implementation
@hook_specification
def on_status_event(experiment_run_id: str, event: StatusEvents) -> None:
    """Called for every status event emitted by a running experiment.

    Implementations are called synchronously before the event is passed on
    to the user, so they must return quickly. The event object is shared, so
    it must not be modified.

    Only one implementation is allowed.

    :param experiment_run_id: ID of the experiment run that emitted the event.
        Reruns of failed realizations reuse the ID, so more than one
        ``EndEvent`` can be received for the same ID. This is not the ID of
        the experiment in storage.
    :param event: The emitted status event.
    """
