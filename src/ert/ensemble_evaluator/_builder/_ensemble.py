import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Union,
)

from cloudevents.conversion import to_json
from cloudevents.http import CloudEvent

from _ert_job_runner.client import Client
from ert.ensemble_evaluator import state
from ert.ensemble_evaluator.snapshot import (
    Job,
    PartialSnapshot,
    RealizationSnapshot,
    Snapshot,
    SnapshotDict,
)
from ert.serialization import evaluator_marshaller

from ._realization import Realization

if TYPE_CHECKING:
    from ..config import EvaluatorServerConfig

logger = logging.getLogger(__name__)

_handle = Callable[..., Any]


class _EnsembleStateTracker:
    def __init__(self, state_: str = state.ENSEMBLE_STATE_UNKNOWN) -> None:
        self._state = state_
        self._handles: Dict[str, _handle] = {}
        self._msg = "Illegal state transition from %s to %s"

        self.set_default_handles()

    def add_handle(self, state_: str, handle: _handle) -> None:
        self._handles[state_] = handle

    def _handle_unknown(self) -> None:
        if self._state != state.ENSEMBLE_STATE_UNKNOWN:
            logger.warning(self._msg, self._state, state.ENSEMBLE_STATE_UNKNOWN)
        self._state = state.ENSEMBLE_STATE_UNKNOWN

    def _handle_started(self) -> None:
        if self._state != state.ENSEMBLE_STATE_UNKNOWN:
            logger.warning(self._msg, self._state, state.ENSEMBLE_STATE_STARTED)
        self._state = state.ENSEMBLE_STATE_STARTED

    def _handle_failed(self) -> None:
        if self._state not in [
            state.ENSEMBLE_STATE_UNKNOWN,
            state.ENSEMBLE_STATE_STARTED,
        ]:
            logger.warning(self._msg, self._state, state.ENSEMBLE_STATE_FAILED)
        self._state = state.ENSEMBLE_STATE_FAILED

    def _handle_stopped(self) -> None:
        if self._state != state.ENSEMBLE_STATE_STARTED:
            logger.warning(self._msg, self._state, state.ENSEMBLE_STATE_STOPPED)
        self._state = state.ENSEMBLE_STATE_STOPPED

    def _handle_canceled(self) -> None:
        if self._state != state.ENSEMBLE_STATE_STARTED:
            logger.warning(self._msg, self._state, state.ENSEMBLE_STATE_CANCELLED)
        self._state = state.ENSEMBLE_STATE_CANCELLED

    def set_default_handles(self) -> None:
        self.add_handle(state.ENSEMBLE_STATE_UNKNOWN, self._handle_unknown)
        self.add_handle(state.ENSEMBLE_STATE_STARTED, self._handle_started)
        self.add_handle(state.ENSEMBLE_STATE_FAILED, self._handle_failed)
        self.add_handle(state.ENSEMBLE_STATE_STOPPED, self._handle_stopped)
        self.add_handle(state.ENSEMBLE_STATE_CANCELLED, self._handle_canceled)

    def update_state(self, state_: str) -> str:
        if state_ not in self._handles:
            raise KeyError(f"Handle not defined for state {state_}")

        # Call the state handle mapped to the new state
        self._handles[state_]()

        return self._state


class Ensemble:
    def __init__(
        self, reals: Sequence[Realization], metadata: Mapping[str, Any], id_: str
    ) -> None:
        self.reals = reals
        self.metadata = metadata
        self._snapshot = self._create_snapshot()
        self.status = self._snapshot.status
        if self._snapshot.status:
            self._status_tracker = _EnsembleStateTracker(self._snapshot.status)
        else:
            self._status_tracker = _EnsembleStateTracker()
        self._id: str = id_

    def __repr__(self) -> str:
        return f"Ensemble with {len(self.reals)} members"

    def evaluate(self, config: "EvaluatorServerConfig") -> None:
        pass

    def cancel(self) -> None:
        pass

    @property
    def id_(self) -> str:
        return self._id

    @property
    def cancellable(self) -> bool:
        return False

    @property
    def active_reals(self) -> Sequence[Realization]:
        return list(filter(lambda real: real.active, self.reals))

    @property
    def snapshot(self) -> Snapshot:
        return self._snapshot

    def update_snapshot(self, events: List[CloudEvent]) -> PartialSnapshot:
        snapshot_mutate_event = PartialSnapshot(self._snapshot)
        for event in events:
            snapshot_mutate_event.from_cloudevent(event)
        self._snapshot.merge_event(snapshot_mutate_event)
        if self._snapshot.status is not None and self.status != self._snapshot.status:
            self.status = self._status_tracker.update_state(self._snapshot.status)
        return snapshot_mutate_event

    async def send_cloudevent(
        self,
        url: str,
        event: CloudEvent,
        token: Optional[str] = None,
        cert: Optional[Union[str, bytes]] = None,
        retries: int = 10,
    ) -> None:
        async with Client(url, token, cert, max_retries=retries) as client:
            await client._send(to_json(event, data_marshaller=evaluator_marshaller))

    def get_successful_realizations(self) -> List[int]:
        return self._snapshot.get_successful_realizations()

    def _create_snapshot(self) -> Snapshot:
        reals: Dict[str, RealizationSnapshot] = {}
        for real in self.active_reals:
            reals[str(real.iens)] = RealizationSnapshot(
                active=True,
                status=state.REALIZATION_STATE_WAITING,
            )
            for index, forward_model in enumerate(real.forward_models):
                reals[str(real.iens)].jobs[str(index)] = Job(
                    status=state.JOB_STATE_START,
                    index=str(index),
                    name=forward_model.name,
                )
        top = SnapshotDict(
            reals=reals,
            status=state.ENSEMBLE_STATE_UNKNOWN,
            metadata=self.metadata,
        )

        return Snapshot(top.dict())
