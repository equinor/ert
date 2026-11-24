from __future__ import annotations

from typing import TYPE_CHECKING

from ert.plugins.plugin_manager import hook_specification, single_implementation

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ert.run_models.event import StatusEvents


@single_implementation
@hook_specification
def status_event_context(config_path: str) -> Mapping[str, str]:  # type: ignore
    """Called once per experiment run when it is registered, before any
    status events of the run are passed to :func:`on_status_event`.

    The returned mapping is passed as ``context`` to every
    :func:`on_status_event` call for the experiment run, also for reruns of
    failed realizations. This lets the plugin attach information about the
    experiment run to its events, without having to look it up for every
    event.

    Implementations are called in a worker thread, and must not block for
    long, as that delays the start of the experiment. Exceptions raised by
    the implementation are logged, and an empty context is used instead.

    Only one implementation is allowed.

    :param config_path: Absolute path of the directory containing the config
        file of the experiment. For Ert this is ``<CONFIG_PATH>``, and for
        Everest it is the directory of the Everest config file.
    :return: The context to pass with every status event of the experiment
        run.
    """


@single_implementation
@hook_specification
def on_status_event(
    experiment_run_id: str, context: Mapping[str, str], event: StatusEvents
) -> None:
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
    :param context: The context returned by :func:`status_event_context` for
        the experiment run, or an empty mapping if that hook is not
        implemented or failed. Implementations that do not need it can leave
        the parameter out.
    :param event: The emitted status event.
    """
