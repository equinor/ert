from __future__ import annotations

import asyncio
import logging
import uuid
from functools import partialmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
)

from cloudevents.http.event import CloudEvent

from _ert.threading import ErtThread
from ert.async_utils import get_event_loop, new_event_loop
from ert.ensemble_evaluator import identifiers
from ert.job_queue import JobQueue
from ert.scheduler import Scheduler, create_driver
from ert.shared.feature_toggling import FeatureScheduler

from .._wait_for_evaluator import wait_for_evaluator
from ._ensemble import Ensemble

if TYPE_CHECKING:
    from ert.config import QueueConfig

    from ..config import EvaluatorServerConfig
    from ._realization import Realization

CONCURRENT_INTERNALIZATION = 10

logger = logging.getLogger(__name__)
event_logger = logging.getLogger("ert.event_log")
scheduler_logger = logging.getLogger("ert.scheduler")


class _KillAllJobs(Protocol):
    def kill_all_jobs(self) -> None: ...


class LegacyEnsemble(Ensemble):
    def __init__(
        self,
        reals: List[Realization],
        metadata: Dict[str, Any],
        queue_config: QueueConfig,
        stop_long_running: bool,
        min_required_realizations: int,
        id_: str,
    ) -> None:
        super().__init__(reals, metadata, id_)

        self._queue_config = queue_config
        self._job_queue: Optional[_KillAllJobs] = None
        self.stop_long_running = stop_long_running
        self.min_required_realizations = min_required_realizations
        self._config: Optional[EvaluatorServerConfig] = None

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
            timeout_queue.put_nowait(
                event_generator(identifiers.EVTYPE_REALIZATION_TIMEOUT, iens)
            )

        async def send_timeout_message() -> None:
            while True:
                timeout_cloudevent = await timeout_queue.get()
                if timeout_cloudevent is None:
                    break
                assert self._config  # mypy
                await cloudevent_unary_send(timeout_cloudevent)

        send_timeout_future = get_event_loop().create_task(send_timeout_message())

        return on_timeout, send_timeout_future

    def evaluate(self, config: EvaluatorServerConfig) -> None:
        if not config:
            raise ValueError("no config for evaluator")
        self._config = config
        get_event_loop().run_until_complete(
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
        get_event_loop().run_until_complete(
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
                scheduler_logger.info("Experiment ran on ORCHESTRATOR: scheduler")
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
                scheduler_logger.info("Experiment ran on ORCHESTRATOR: job_queue")
            self._job_queue = queue

            await cloudevent_unary_send(
                event_creator(identifiers.EVTYPE_ENSEMBLE_STARTED, None)
            )

            min_required_realizations = (
                self.min_required_realizations if self.stop_long_running else 0
            )

            queue.add_dispatch_information_to_jobs_file()
            result = await queue.execute(min_required_realizations)

        except Exception:
            logger.exception(
                "unexpected exception in ensemble",
                exc_info=True,
            )
            result = identifiers.EVTYPE_ENSEMBLE_FAILED

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
