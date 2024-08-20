from __future__ import annotations

import asyncio
import logging
import traceback
import uuid
from dataclasses import dataclass
from functools import partialmethod
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Union,
)

from cloudevents.conversion import to_json
from cloudevents.http import CloudEvent

from _ert_forward_model_runner.client import Client
from ert.config import ForwardModelStep, QueueConfig
from ert.run_arg import RunArg
from ert.scheduler import Scheduler, create_driver
from ert.serialization import evaluator_marshaller

from ._wait_for_evaluator import wait_for_evaluator
from .config import EvaluatorServerConfig
from .identifiers import EVTYPE_ENSEMBLE_FAILED, EVTYPE_ENSEMBLE_STARTED
from .snapshot import ForwardModel, RealizationSnapshot, Snapshot, SnapshotDict
from .state import (
    ENSEMBLE_STATE_CANCELLED,
    ENSEMBLE_STATE_FAILED,
    ENSEMBLE_STATE_STARTED,
    ENSEMBLE_STATE_STOPPED,
    ENSEMBLE_STATE_UNKNOWN,
    FORWARD_MODEL_STATE_START,
    REALIZATION_STATE_WAITING,
)

logger = logging.getLogger(__name__)

_handle = Callable[..., Any]


class _EnsembleStateTracker:
    def __init__(self, state_: str = ENSEMBLE_STATE_UNKNOWN) -> None:
        self._state = state_
        self._handles: Dict[str, _handle] = {}
        self._msg = "Illegal state transition from %s to %s"

        self.set_default_handles()

    def add_handle(self, state_: str, handle: _handle) -> None:
        self._handles[state_] = handle

    def _handle_unknown(self) -> None:
        if self._state != ENSEMBLE_STATE_UNKNOWN:
            logger.warning(self._msg, self._state, ENSEMBLE_STATE_UNKNOWN)
        self._state = ENSEMBLE_STATE_UNKNOWN

    def _handle_started(self) -> None:
        if self._state != ENSEMBLE_STATE_UNKNOWN:
            logger.warning(self._msg, self._state, ENSEMBLE_STATE_STARTED)
        self._state = ENSEMBLE_STATE_STARTED

    def _handle_failed(self) -> None:
        if self._state not in [
            ENSEMBLE_STATE_UNKNOWN,
            ENSEMBLE_STATE_STARTED,
        ]:
            logger.warning(self._msg, self._state, ENSEMBLE_STATE_FAILED)
        self._state = ENSEMBLE_STATE_FAILED

    def _handle_stopped(self) -> None:
        if self._state != ENSEMBLE_STATE_STARTED:
            logger.warning(self._msg, self._state, ENSEMBLE_STATE_STOPPED)
        self._state = ENSEMBLE_STATE_STOPPED

    def _handle_canceled(self) -> None:
        if self._state != ENSEMBLE_STATE_STARTED:
            logger.warning(self._msg, self._state, ENSEMBLE_STATE_CANCELLED)
        self._state = ENSEMBLE_STATE_CANCELLED

    def set_default_handles(self) -> None:
        self.add_handle(ENSEMBLE_STATE_UNKNOWN, self._handle_unknown)
        self.add_handle(ENSEMBLE_STATE_STARTED, self._handle_started)
        self.add_handle(ENSEMBLE_STATE_FAILED, self._handle_failed)
        self.add_handle(ENSEMBLE_STATE_STOPPED, self._handle_stopped)
        self.add_handle(ENSEMBLE_STATE_CANCELLED, self._handle_canceled)

    def update_state(self, state_: str) -> str:
        if state_ not in self._handles:
            raise KeyError(f"Handle not defined for state {state_}")

        # Call the state handle mapped to the new state
        self._handles[state_]()

        return self._state


@dataclass
class LegacyEnsemble:
    reals: List[Realization]
    metadata: Dict[str, Any]
    _queue_config: QueueConfig
    min_required_realizations: int
    id_: str

    def __post_init__(self) -> None:
        self._scheduler: Optional[_KillAllJobs] = None
        self._config: Optional[EvaluatorServerConfig] = None
        self.snapshot: Snapshot = self._create_snapshot()
        self.status = self.snapshot.status
        if self.snapshot.status:
            self._status_tracker = _EnsembleStateTracker(self.snapshot.status)
        else:
            self._status_tracker = _EnsembleStateTracker()

    @property
    def active_reals(self) -> Sequence[Realization]:
        return list(filter(lambda real: real.active, self.reals))

    def _create_snapshot(self) -> Snapshot:
        reals: Dict[str, RealizationSnapshot] = {}
        for real in self.active_reals:
            reals[str(real.iens)] = RealizationSnapshot(
                active=True,
                status=REALIZATION_STATE_WAITING,
            )
            for index, forward_model in enumerate(real.forward_models):
                reals[str(real.iens)].forward_models[str(index)] = ForwardModel(
                    status=FORWARD_MODEL_STATE_START,
                    index=str(index),
                    name=forward_model.name,
                )
        top = SnapshotDict(
            reals=reals,
            status=ENSEMBLE_STATE_UNKNOWN,
            metadata=self.metadata,
        )

        return Snapshot.from_nested_dict(top.model_dump())

    def get_successful_realizations(self) -> List[int]:
        return self.snapshot.get_successful_realizations()

    def update_snapshot(self, events: List[CloudEvent]) -> Snapshot:
        snapshot_mutate_event = Snapshot()
        for event in events:
            snapshot_mutate_event = snapshot_mutate_event.update_from_cloudevent(
                event, source_snapshot=self.snapshot
            )
        self.snapshot.merge_snapshot(snapshot_mutate_event)
        if self.snapshot.status is not None and self.status != self.snapshot.status:
            self.status = self._status_tracker.update_state(self.snapshot.status)
        return snapshot_mutate_event

    async def send_cloudevent(  # noqa: PLR6301
        self,
        url: str,
        event: CloudEvent,
        token: Optional[str] = None,
        cert: Optional[Union[str, bytes]] = None,
        retries: int = 10,
    ) -> None:
        async with Client(url, token, cert, max_retries=retries) as client:
            await client._send(to_json(event, data_marshaller=evaluator_marshaller))

    def generate_event_creator(self) -> Callable[[str, Optional[int]], CloudEvent]:
        def event_builder(status: str, real_id: Optional[int] = None) -> CloudEvent:
            source = f"/ert/ensemble/{self.id_}"
            if real_id is not None:
                source += f"/real/{real_id}"
            return CloudEvent(
                {
                    "type": status,
                    "source": source,
                    "id": str(uuid.uuid1()),
                }
            )

        return event_builder

    async def evaluate(
        self,
        config: EvaluatorServerConfig,
        scheduler_queue: Optional[asyncio.Queue[CloudEvent]] = None,
        manifest_queue: Optional[asyncio.Queue[CloudEvent]] = None,
    ) -> None:
        self._config = config
        ce_unary_send_method_name = "_ce_unary_send"
        setattr(
            self.__class__,
            ce_unary_send_method_name,
            partialmethod(
                self.__class__.send_cloudevent,
                self._config.dispatch_uri,
                token=self._config.token,
                cert=self._config.cert,
            ),
        )
        await wait_for_evaluator(
            base_url=self._config.url,
            token=self._config.token,
            cert=self._config.cert,
        )
        await self._evaluate_inner(
            cloudevent_unary_send=getattr(self, ce_unary_send_method_name),
            scheduler_queue=scheduler_queue,
            manifest_queue=manifest_queue,
        )

    async def _evaluate_inner(  # pylint: disable=too-many-branches
        self,
        cloudevent_unary_send: Callable[[CloudEvent], Awaitable[None]],
        scheduler_queue: Optional[asyncio.Queue[CloudEvent]] = None,
        manifest_queue: Optional[asyncio.Queue[CloudEvent]] = None,
    ) -> None:
        """
        This (inner) coroutine does the actual work of evaluating the ensemble. It
        prepares and executes the necessary bookkeeping, prepares and executes
        the JobQueue, and dispatches pertinent events.

        Before returning, it always dispatches a CloudEvent describing
        the final result of executing all its jobs through a JobQueue.

        cloudevent_unary_send determines how CloudEvents are dispatched. This
        is a function (or bound method) that only takes a CloudEvent as a positional
        argument.
        """
        event_creator = self.generate_event_creator()

        if not self.id_:
            raise ValueError("Ensemble id not set")
        if not self._config:
            raise ValueError("no config")  # mypy

        try:
            driver = create_driver(self._queue_config)
            self._scheduler = Scheduler(
                driver,
                self.active_reals,
                manifest_queue,
                scheduler_queue,
                max_submit=self._queue_config.max_submit,
                max_running=self._queue_config.max_running,
                submit_sleep=self._queue_config.submit_sleep,
                ens_id=self.id_,
                ee_uri=self._config.dispatch_uri,
                ee_cert=self._config.cert,
                ee_token=self._config.token,
            )
            logger.info(
                f"Experiment ran on ORCHESTRATOR: scheduler on {self._queue_config.queue_system} queue"
            )

            await cloudevent_unary_send(event_creator(EVTYPE_ENSEMBLE_STARTED, None))

            min_required_realizations = (
                self.min_required_realizations
                if self._queue_config.stop_long_running
                else 0
            )

            self._scheduler.add_dispatch_information_to_jobs_file()
            result = await self._scheduler.execute(min_required_realizations)

        except Exception as exc:
            logger.exception(
                (
                    "Unexpected exception in ensemble: \n" "".join(
                        traceback.format_exception(None, exc, exc.__traceback__)
                    )
                ),
                exc_info=True,
            )
            await cloudevent_unary_send(event_creator(EVTYPE_ENSEMBLE_FAILED, None))
            return

        logger.info(f"Experiment ran on QUEUESYSTEM: {self._queue_config.queue_system}")

        # Dispatch final result from evaluator - FAILED, CANCEL or STOPPED
        await cloudevent_unary_send(event_creator(result, None))

    @property
    def cancellable(self) -> bool:
        return True

    def cancel(self) -> None:
        if self._scheduler is not None:
            self._scheduler.kill_all_jobs()
        logger.debug("evaluator cancelled")


class _KillAllJobs(Protocol):
    def kill_all_jobs(self) -> None: ...


@dataclass
class Realization:
    iens: int
    forward_models: Sequence[ForwardModelStep]
    active: bool
    max_runtime: Optional[int]
    run_arg: "RunArg"
    num_cpu: int
    job_script: str
    realization_memory: int  # Memory to reserve/book, in bytes
