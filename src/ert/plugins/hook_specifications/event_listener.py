from __future__ import annotations

from typing import TYPE_CHECKING

from ert.plugins.plugin_manager import hook_specification, single_implementation

if TYPE_CHECKING:
    from ert.run_models.event import StatusEvents


@single_implementation
@hook_specification
def on_status_event(experiment_run_id: str, event: StatusEvents) -> None:
    """Called for every status event emitted by a running experiment.

    Implementations are called in a worker thread, without delaying the
    experiment. The events of each experiment run are passed one at a time
    in the order they were emitted, also across reruns of failed
    realizations. Events from different experiment runs can be passed
    concurrently from different threads. The event is a copy, so modifying
    it does not affect other subscribers.

    Only one implementation is allowed.

    Implementations must not block indefinitely, as a call that never
    returns blocks all later events for its experiment run, can delay
    events for other experiment runs and can prevent the server from
    shutting down.

    :param experiment_run_id: ID of the experiment run that emitted the event.
        Reruns of failed realizations reuse the ID, so more than one
        ``EndEvent`` can be received for the same ID. This is not the ID of
        the experiment in storage.
    :param event: The emitted status event.
    """
