from __future__ import annotations

from typing import TYPE_CHECKING

from ert.plugins.plugin_manager import hook_specification, single_implementation

if TYPE_CHECKING:
    from ert.run_models.event import StatusEvents


@single_implementation
@hook_specification
def on_status_event(experiment_run_id: str, event: StatusEvents) -> None:
    """Called for every status event emitted by a running experiment.

    Implementations are called in a worker thread, so they do not delay the
    experiment, but may receive events later than they were emitted. The
    events of each experiment run are passed one at a time, in the order
    they were emitted, also across reruns of failed realizations. Events
    from different experiment runs can be passed concurrently from different
    threads, so state shared between runs must be protected, for example
    with a lock.

    The event is a copy, so modifying it does not affect Ert. Exceptions
    raised by the implementation are logged, and later events are still
    passed.

    Implementations must not block indefinitely, as that blocks all later
    events for the experiment run, and can delay events for other
    experiment runs and the shutdown of the server.

    Only one implementation is allowed.

    :param experiment_run_id: UUID identifying the experiment run that
        emitted the event. Reruns of failed realizations reuse the ID. This
        is not the ID of the experiment in storage.
    :param event: The emitted status event.
    """
