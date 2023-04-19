from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Union
from uuid import UUID

from cloudevents.http import CloudEvent

if TYPE_CHECKING:
    from _ert_com_protocol import DispatcherMessage
    from ert.ensemble_evaluator import EvaluatorServerConfig


class Experiment(Protocol):
    """The experiment protocol which all experiments must implement."""

    async def run(self, evaluator_server_config: EvaluatorServerConfig) -> None:
        """Run the experiment to completion."""

    @property
    def id(self) -> UUID:
        """The id of the experiment."""

    async def dispatch(self, event: Union[CloudEvent, DispatcherMessage]) -> None:
        """dispatch(self, event) -> None
        event is a ``CloudEvent`` https://github.com/cloudevents/sdk-python
        or a ``protocol buffer object`` https://developers.google.com/protocol-buffers
        The experiment will internalize the event and update its state.
        """

    # TODO: this is preliminary, see https://github.com/equinor/ert/issues/3407
    async def successful_realizations(self, iter_: int) -> int:
        """Return the amount of successful realizations."""
