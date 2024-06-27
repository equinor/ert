from __future__ import annotations

import asyncio
import logging
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
    Tuple,
    Union,
)

from cloudevents.conversion import to_json
from cloudevents.http import CloudEvent

from _ert.async_utils import get_running_loop, new_event_loop
from _ert.threading import ErtThread
from _ert_forward_model_runner.client import Client
from ert.config import ForwardModelStep, QueueConfig
from ert.job_queue import JobQueue
from ert.run_arg import RunArg
from ert.scheduler import Scheduler, create_driver
from ert.serialization import evaluator_marshaller
from ert.shared.feature_toggling import FeatureScheduler

from ._wait_for_evaluator import wait_for_evaluator
from .config import EvaluatorServerConfig
from .identifiers import (
    EVTYPE_ENSEMBLE_FAILED,
    EVTYPE_ENSEMBLE_STARTED,
    EVTYPE_REALIZATION_TIMEOUT,
)
from .snapshot import (
    ForwardModel,
    PartialSnapshot,
    RealizationSnapshot,
    Snapshot,
    SnapshotDict,
)
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
scheduler_logger = logging.getLogger("ert.scheduler")

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
        self._job_queue: Optional[_KillAllJobs] = None
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

        return Snapshot(top.model_dump())

    def get_successful_realizations(self) -> List[int]:
        return self.snapshot.get_successful_realizations()

    def update_snapshot(self, events: List[CloudEvent]) -> PartialSnapshot:
        snapshot_mutate_event = PartialSnapshot(self.snapshot)
        for event in events:
            snapshot_mutate_event.from_cloudevent(event)
        self.snapshot.merge_event(snapshot_mutate_event)
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

    def generate_event_creator(
        self, experiment_id: Optional[str] = None
    ) -> Callable[[str, Optional[int]], CloudEvent]:
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

    def setup_timeout_callback(
        self,
        timeout_queue: asyncio.Queue[CloudEvent],
        cloudevent_unary_send: Callable[[CloudEvent], Awaitable[None]],
        event_generator: Callable[[str, Optional[int]], CloudEvent],
    ) -> Tuple[Callable[[int], None], asyncio.Task[None]]:
        """This function is reimplemented inside the Scheduler and should
        be removed when Scheduler is the only queue code."""

        def on_timeout(iens: int) -> None:
            timeout_queue.put_nowait(event_generator(EVTYPE_REALIZATION_TIMEOUT, iens))

        async def send_timeout_message() -> None:
            while True:
                timeout_cloudevent = await timeout_queue.get()
                if timeout_cloudevent is None:
                    break
                assert self._config  # mypy
                await cloudevent_unary_send(timeout_cloudevent)

        send_timeout_future = get_running_loop().create_task(send_timeout_message())

        return on_timeout, send_timeout_future

    def evaluate(self, config: EvaluatorServerConfig) -> None:
        if not config:
            raise ValueError("no config for evaluator")
        self._config = config
        get_running_loop().run_until_complete(
            wait_for_evaluator(
                base_url=self._config.url,
                token=self._config.token,
                cert=self._config.cert,
            )
        )

        ErtThread(target=self._evaluate, name="LegacyEnsemble").start()

    def _evaluate(self) -> None:
        """
        This method is executed on a separate thread, i.e. in parallel
        with other threads. Its sole purpose is to execute and wait for
        a coroutine
        """
        # Get a fresh eventloop
        asyncio.set_event_loop(new_event_loop())

        if self._config is None:
            raise ValueError("no config")

        # The cloudevent_unary_send only accepts a cloud event, but in order to
        # send cloud events over the network, we need token, URI and cert. These are
        # not known until evaluate() is called and _config is set. So in a hacky
        # fashion, we create the partialmethod (bound partial) here, after evaluate().
        # Note that this is the "sync" version of evaluate(), and that the "async"
        # version uses a different cloudevent_unary_send.
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
        get_running_loop().run_until_complete(
            self._evaluate_inner(
                cloudevent_unary_send=getattr(self, ce_unary_send_method_name)
            )
        )

    async def _evaluate_inner(  # pylint: disable=too-many-branches
        self,
        cloudevent_unary_send: Callable[[CloudEvent], Awaitable[None]],
        experiment_id: Optional[str] = None,
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
        event_creator = self.generate_event_creator(experiment_id=experiment_id)
        timeout_queue: Optional[asyncio.Queue[Any]] = None
        using_scheduler = FeatureScheduler.is_enabled(self._queue_config.queue_system)

        if not using_scheduler:
            # Set up the timeout-mechanism
            timeout_queue = asyncio.Queue()
            # Based on the experiment id the generator will
            # give a function returning cloud event
            on_timeout, send_timeout_future = self.setup_timeout_callback(
                timeout_queue, cloudevent_unary_send, event_creator
            )

        if not self.id_:
            raise ValueError("Ensemble id not set")
        if not self._config:
            raise ValueError("no config")  # mypy

        try:
            if using_scheduler:
                driver = create_driver(self._queue_config)
                queue = Scheduler(
                    driver,
                    self.active_reals,
                    max_submit=self._queue_config.max_submit,
                    max_running=self._queue_config.max_running,
                    submit_sleep=self._queue_config.submit_sleep,
                    ens_id=self.id_,
                    ee_uri=self._config.dispatch_uri,
                    ee_cert=self._config.cert,
                    ee_token=self._config.token,
                )
                scheduler_logger.info(
                    f"Experiment ran on ORCHESTRATOR: scheduler on {self._queue_config.queue_system} queue"
                )
            else:
                queue = JobQueue(
                    self._queue_config,
                    self.active_reals,
                    ens_id=self.id_,
                    ee_uri=self._config.dispatch_uri,
                    ee_cert=self._config.cert,
                    ee_token=self._config.token,
                    on_timeout=on_timeout,
                )
                scheduler_logger.info(
                    f"Experiment ran on ORCHESTRATOR: job_queue on {self._queue_config.queue_system}"
                )
            self._job_queue = queue

            await cloudevent_unary_send(event_creator(EVTYPE_ENSEMBLE_STARTED, None))

            min_required_realizations = (
                self.min_required_realizations
                if self._queue_config.stop_long_running
                else 0
            )

            queue.add_dispatch_information_to_jobs_file()
            result = await queue.execute(min_required_realizations)

        except Exception:
            logger.exception(
                "unexpected exception in ensemble",
                exc_info=True,
            )
            result = EVTYPE_ENSEMBLE_FAILED

        if not isinstance(self._job_queue, Scheduler):
            assert timeout_queue is not None
            await timeout_queue.put(None)  # signal to exit timer
            await send_timeout_future

        scheduler_logger.info(
            f"Experiment ran on QUEUESYSTEM: {self._queue_config.queue_system}"
        )

        # Dispatch final result from evaluator - FAILED, CANCEL or STOPPED
        await cloudevent_unary_send(event_creator(result, None))

    @property
    def cancellable(self) -> bool:
        return True

    def cancel(self) -> None:
        if self._job_queue is not None:
            self._job_queue.kill_all_jobs()
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
