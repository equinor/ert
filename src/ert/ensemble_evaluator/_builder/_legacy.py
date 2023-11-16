from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from functools import partial, partialmethod
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, List, Optional, Tuple

from cloudevents.http.event import CloudEvent

from ert.async_utils import get_event_loop
from ert.ensemble_evaluator import identifiers
from ert.job_queue import JobQueue

from .._wait_for_evaluator import wait_for_evaluator
from ._ensemble import Ensemble

if TYPE_CHECKING:
    from ert.config import QueueConfig

    from ..config import EvaluatorServerConfig
    from ._realization import Realization

CONCURRENT_INTERNALIZATION = 10

logger = logging.getLogger(__name__)
event_logger = logging.getLogger("ert.event_log")


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
        if not queue_config:
            raise ValueError(f"{self} needs queue_config")
        self._job_queue = JobQueue(queue_config)
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

        threading.Thread(target=self._evaluate, name="LegacyEnsemble").start()

    def _evaluate(self) -> None:
        """
        This method is executed on a separate thread, i.e. in parallel
        with other threads. Its sole purpose is to execute and wait for
        a coroutine
        """
        # Get a fresh eventloop
        asyncio.set_event_loop(asyncio.new_event_loop())

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
        try:
            get_event_loop().run_until_complete(
                self._evaluate_inner(
                    cloudevent_unary_send=getattr(self, ce_unary_send_method_name)
                )
            )
        finally:
            get_event_loop().close()

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
        # Set up the timeout-mechanism
        timeout_queue = asyncio.Queue()  # type: ignore
        # Based on the experiment id the generator will
        # give a function returning cloud event
        event_creator = self.generate_event_creator(experiment_id=experiment_id)
        on_timeout, send_timeout_future = self.setup_timeout_callback(
            timeout_queue, cloudevent_unary_send, event_creator
        )

        if not self.id_:
            raise ValueError("Ensemble id not set")
        if not self._config:
            raise ValueError("no config")  # mypy

        try:
            await cloudevent_unary_send(
                event_creator(identifiers.EVTYPE_ENSEMBLE_STARTED, None)
            )

            for real in self.active_reals:
                self._job_queue.add_realization(real, callback_timeout=on_timeout)

            # TODO: this is sort of a callback being preemptively called.
            # It should be lifted out of the queue/evaluate, into the evaluator. If
            # something is long running, the evaluator will know and should send
            # commands to the task in order to have it killed/retried.
            # See https://github.com/equinor/ert/issues/1229
            queue_evaluators = None
            if self.stop_long_running and self.min_required_realizations > 0:
                queue_evaluators = [
                    partial(
                        self._job_queue.stop_long_running_jobs,
                        self.min_required_realizations,
                    )
                ]

            self._job_queue.set_ee_info(
                ee_uri=self._config.dispatch_uri,
                ens_id=self.id_,
                ee_cert=self._config.cert,
                ee_token=self._config.token,
            )

            # Tell queue to pass info to the jobs-file
            # NOTE: This touches files on disk...
            self._job_queue.add_dispatch_information_to_jobs_file()

            sema = threading.BoundedSemaphore(value=CONCURRENT_INTERNALIZATION)
            result: str = await self._job_queue.execute(
                sema,
                queue_evaluators,
            )

        except Exception:
            logger.exception(
                "unexpected exception in ensemble",
                exc_info=True,
            )
            result = identifiers.EVTYPE_ENSEMBLE_FAILED

        await timeout_queue.put(None)  # signal to exit timer
        await send_timeout_future

        # Dispatch final result from evaluator - FAILED, CANCEL or STOPPED
        await cloudevent_unary_send(event_creator(result, None))

    @property
    def cancellable(self) -> bool:
        return True

    def cancel(self) -> None:
        self._job_queue.kill_all_jobs()
        logger.debug("evaluator cancelled")
