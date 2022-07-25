from typing import TYPE_CHECKING
from typing_extensions import Protocol
from cloudevents.http import CloudEvent

if TYPE_CHECKING:
    from ert_shared.ensemble_evaluator.config import EvaluatorServerConfig


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

    async def dispatch(self, event: CloudEvent, iter_: int) -> None:
        """dispatch(self, event, iter_: int) -> None

        event is a ``CloudEvent`` https://github.com/cloudevents/sdk-python

        Pass an event for ``iter_`` to the experiment. The experiment will internalize
        the event and update its state."""
        pass

    # TODO: this is preliminary, see https://github.com/equinor/ert/issues/3407
    async def successful_realizations(self, iter_: int) -> int:
        """Return the amount of successful realizations."""
        pass
