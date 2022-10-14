from typing import TYPE_CHECKING, Union

from cloudevents.http import CloudEvent
from typing_extensions import Protocol

from _ert_com_protocol import DispatcherMessage

if TYPE_CHECKING:
    from ert.ensemble_evaluator import EvaluatorServerConfig


class Experiment(Protocol):
    """The experiment protocol which all experiments must implement."""

    async def run(self, evaluator_server_config: "EvaluatorServerConfig") -> None:
        """Run the experiment to completion."""
        pass

    @property
    def id_(self) -> str:
        """The id of the experiment."""
        pass

    @id_.setter
    def id_(self, value: str) -> None:
        """Set the of the experiment to ``value``. It should not be possible
        to set this more than once."""
        pass

    async def dispatch(self, event: Union[CloudEvent, DispatcherMessage]) -> None:
        """dispatch(self, event) -> None
        event is a ``CloudEvent`` https://github.com/cloudevents/sdk-python
        or a ``protocol buffer object`` https://developers.google.com/protocol-buffers
        The experiment will internalize the event and update its state.
        """
        pass

    # TODO: this is preliminary, see https://github.com/equinor/ert/issues/3407
    async def successful_realizations(self, iter_: int) -> int:
        """Return the amount of successful realizations."""
        pass
