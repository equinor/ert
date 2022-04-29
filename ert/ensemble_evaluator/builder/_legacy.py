import asyncio
import logging
import threading
import uuid
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

from cloudevents.http.event import CloudEvent
from ert.ensemble_evaluator import identifiers
from ert.ensemble_evaluator.util._network import wait_for_evaluator
from ert_shared.async_utils import get_event_loop
from res.enkf import RunArg
from res.enkf.analysis_config import AnalysisConfig
from res.enkf.queue_config import QueueConfig

from ._ensemble import _Ensemble
from ._realization import _Realization

if TYPE_CHECKING:
    from ert_shared.ensemble_evaluator.config import EvaluatorServerConfig

CONCURRENT_INTERNALIZATION = 10

logger = logging.getLogger(__name__)


class _LegacyEnsemble(_Ensemble):
    def __init__(
        self,
        reals: List[_Realization],
        metadata: Dict[str, Any],
        queue_config: QueueConfig,
        analysis_config: AnalysisConfig,
    ) -> None:
        super().__init__(reals, metadata)
        if not queue_config:
            raise ValueError(f"{self} needs queue_config")
        if not analysis_config:
            raise ValueError(f"{self} needs analysis_config")
        self._job_queue = queue_config.create_job_queue()
        self._analysis_config = analysis_config
        self._config: Optional["EvaluatorServerConfig"] = None
        self._ee_id: Optional[str] = None

    def setup_timeout_callback(
        self, timeout_queue: "asyncio.Queue[CloudEvent]"
    ) -> Tuple[Callable[[List[Any]], Any], "asyncio.Task[None]"]:
        def on_timeout(callback_args: Sequence[Any]) -> None:
            run_args: RunArg = callback_args[0]
            timeout_cloudevent = CloudEvent(
                {
                    "type": identifiers.EVTYPE_FM_STEP_TIMEOUT,
                    "source": f"/ert/ee/{self._ee_id}/real/{run_args.iens}/step/0",
                    "id": str(uuid.uuid1()),
                }
            )
            timeout_queue.put_nowait(timeout_cloudevent)

        async def send_timeout_message() -> None:
            while True:
                timeout_cloudevent = await timeout_queue.get()
                if timeout_cloudevent is None:
                    break
                assert self._config  # mypy
                await self.send_cloudevent(
                    self._config.dispatch_uri,
                    timeout_cloudevent,
                    token=self._config.token,
                    cert=self._config.cert,
                )

        send_timeout_future = get_event_loop().create_task(send_timeout_message())

        return on_timeout, send_timeout_future

    def evaluate(self, config: "EvaluatorServerConfig", ee_id: str) -> None:
        if not config:
            raise ValueError("no config for evaluator {ee_id}")
        self._config = config
        self._ee_id = ee_id
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

        async def _evaluate_inner() -> None:
            """
            This (inner) coroutine does the actual work. It prepares and
            executes the necessary bookkeeping, prepares and executes
            the JobQueue, and dispatches pertinent events.

            Before returning, it always dispatches a CloudEvent describing
            the final result of executing all its jobs through a JobQueue.
            """
            # Set up the timeout-mechanism
            timeout_queue: "asyncio.Queue[CloudEvent]" = asyncio.Queue()
            on_timeout, send_timeout_future = self.setup_timeout_callback(timeout_queue)

            if not self._ee_id:
                raise ValueError(f"invalid ensemble evaluator id: {self._ee_id}")
            if not self._config:
                raise ValueError("no config")  # mypy

            # event for normal evaluation
            result = CloudEvent(
                {
                    "type": identifiers.EVTYPE_ENSEMBLE_STOPPED,
                    "source": f"/ert/ee/{self._ee_id}/ensemble",
                    "id": str(uuid.uuid1()),
                }
            )
            try:
                # Dispatch STARTED-event
                out_cloudevent = CloudEvent(
                    {
                        "type": identifiers.EVTYPE_ENSEMBLE_STARTED,
                        "source": f"/ert/ee/{self._ee_id}/ensemble",
                        "id": str(uuid.uuid1()),
                    }
                )
                await self.send_cloudevent(
                    self._config.dispatch_uri,
                    out_cloudevent,
                    token=self._config.token,
                    cert=self._config.cert,
                )

                # Submit all jobs to queue and inform queue when done
                for real in self.active_reals:
                    self._job_queue.add_ee_stage(  # type: ignore
                        real.steps[0], callback_timeout=on_timeout
                    )
                self._job_queue.submit_complete()  # type: ignore

                # TODO: this is sort of a callback being preemptively called.
                # It should be lifted out of the queue/evaluate, into the evaluator. If
                # something is long running, the evaluator will know and should send
                # commands to the task in order to have it killed/retried.
                # See https://github.com/equinor/ert/issues/1229
                queue_evaluators = None
                if (
                    self._analysis_config.get_stop_long_running()
                    and self._analysis_config.minimum_required_realizations > 0
                ):
                    queue_evaluators = [
                        partial(
                            self._job_queue.stop_long_running_jobs,
                            self._analysis_config.minimum_required_realizations,
                        )
                    ]

                # Tell queue to pass info to the jobs-file
                # NOTE: This touches files on disk...
                self._job_queue.add_ensemble_evaluator_information_to_jobs_file(
                    self._ee_id,
                    self._config.dispatch_uri,
                    self._config.cert,
                    self._config.token,
                )

                # Finally, run the queue-loop until it finishes or raises
                await self._job_queue.execute_queue_async(
                    self._config.dispatch_uri,
                    self._ee_id,
                    threading.BoundedSemaphore(value=CONCURRENT_INTERNALIZATION),
                    queue_evaluators,  # type: ignore
                    cert=self._config.cert,
                    token=self._config.token,
                )

            except asyncio.CancelledError:
                logger.debug("ensemble was cancelled")
                result = CloudEvent(
                    {
                        "type": identifiers.EVTYPE_ENSEMBLE_CANCELLED,
                        "source": f"/ert/ee/{self._ee_id}/ensemble",
                        "id": str(uuid.uuid1()),
                    }
                )

            except Exception:  # pylint: disable=broad-except
                logger.exception(
                    "unexpected exception in ensemble",
                    exc_info=True,
                )
                result = CloudEvent(
                    {
                        "type": identifiers.EVTYPE_ENSEMBLE_FAILED,
                        "source": f"/ert/ee/{self._ee_id}/ensemble",
                        "id": str(uuid.uuid1()),
                    }
                )

            else:
                logger.debug("ensemble finished normally")

            finally:
                await timeout_queue.put(None)  # signal to exit timer
                await send_timeout_future

                # Dispatch final result from evaluator - FAILED, CANCEL or STOPPED
                assert self._config  # mypy
                await self.send_cloudevent(
                    self._config.dispatch_uri,
                    result,
                    token=self._config.token,
                    cert=self._config.cert,
                )

        get_event_loop().run_until_complete(_evaluate_inner())
        get_event_loop().close()

    @property
    def cancellable(self) -> bool:
        return True

    def cancel(self) -> None:
        self._job_queue.kill_all_jobs()
        logger.debug("evaluator cancelled")
