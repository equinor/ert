import asyncio
import logging
import threading
import uuid
from functools import partial, partialmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from cloudevents.http.event import CloudEvent

import _ert_com_protocol
from ert._c_wrappers.enkf import RunArg
from ert._c_wrappers.enkf.analysis_config import AnalysisConfig
from ert._c_wrappers.enkf.queue_config import QueueConfig
from ert.async_utils import get_event_loop
from ert.ensemble_evaluator import identifiers

from .._wait_for_evaluator import wait_for_evaluator
from ._ensemble import Ensemble

if TYPE_CHECKING:
    from ._realization import Realization
    from .config import EvaluatorServerConfig


CONCURRENT_INTERNALIZATION = 10

logger = logging.getLogger(__name__)
event_logger = logging.getLogger("ert.event_log")
experiment_logger = logging.getLogger("ert.experiment_server")


class LegacyEnsemble(Ensemble):
    def __init__(  # pylint: disable=too-many-arguments
        self,
        reals: List["Realization"],
        metadata: Dict[str, Any],
        queue_config: QueueConfig,
        analysis_config: AnalysisConfig,
        id_: str,
    ) -> None:
        super().__init__(reals, metadata, id_)
        if not queue_config:
            raise ValueError(f"{self} needs queue_config")
        if not analysis_config:
            raise ValueError(f"{self} needs analysis_config")
        self._job_queue = queue_config.create_job_queue()
        self._analysis_config = analysis_config
        self._config: Optional["EvaluatorServerConfig"] = None
        self._output_bus: "asyncio.Queue[_ert_com_protocol.DispatcherMessage]" = (
            asyncio.Queue()
        )

    def generate_event_creator(
        self, experiment_id: Optional[str] = None
    ) -> Callable[
        [str, Optional[int]], Union[CloudEvent, _ert_com_protocol.DispatcherMessage]
    ]:
        if experiment_id is not None:

            def node_builder(
                status: str, real_id: Optional[int] = None
            ) -> _ert_com_protocol.DispatcherMessage:
                assert experiment_id  # mypy error
                # TODO: this might got to _ert_com_protocol enum
                status_tab = {
                    identifiers.EVTYPE_ENSEMBLE_STARTED: "ENSEMBLE_STARTED",
                    identifiers.EVTYPE_ENSEMBLE_FAILED: "ENSEMBLE_FAILED",
                    identifiers.EVTYPE_ENSEMBLE_CANCELLED: "ENSEMBLE_CANCELLED",
                    identifiers.EVTYPE_ENSEMBLE_STOPPED: "ENSEMBLE_STOPPED",
                    identifiers.EVTYPE_FM_STEP_TIMEOUT: "STEP_TIMEOUT",
                }
                step_id = None
                if real_id is not None:
                    step_id = 0

                return _ert_com_protocol.node_status_builder(
                    ensemble_id=self.id_,
                    experiment_id=experiment_id,
                    status=status_tab[status],
                    realization_id=real_id,
                    step_id=step_id,
                )

            return node_builder

        else:

            def event_builder(status: str, real_id: Optional[int] = None) -> CloudEvent:
                source = f"/ert/ensemble/{self.id_}"
                if real_id is not None:
                    source += f"/real/{real_id}/step/0"
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
        timeout_queue: "asyncio.Queue[Union[CloudEvent, _ert_com_protocol.DispatcherMessage]]",  # noqa: E501 # pylint: disable=line-too-long
        cloudevent_unary_send: Callable[
            [Union[CloudEvent, _ert_com_protocol.DispatcherMessage]], Awaitable[None]
        ],
        event_generator: Callable[
            [str, Optional[int]], Union[CloudEvent, _ert_com_protocol.DispatcherMessage]
        ],
    ) -> Tuple[Callable[[List[Any]], Any], "asyncio.Task[None]"]:
        def on_timeout(callback_args: Sequence[Any]) -> None:
            run_args: RunArg = callback_args[0]
            timeout_queue.put_nowait(
                event_generator(identifiers.EVTYPE_FM_STEP_TIMEOUT, run_args.iens)
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

    def evaluate(self, config: "EvaluatorServerConfig") -> None:
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
        self.__class__._ce_unary_send = partialmethod(  # type: ignore
            self.__class__.send_cloudevent,
            self._config.dispatch_uri,
            token=self._config.token,
            cert=self._config.cert,
        )
        try:
            get_event_loop().run_until_complete(
                self._evaluate_inner(
                    cloudevent_unary_send=self._ce_unary_send  # type: ignore
                )
            )
        finally:
            get_event_loop().close()

    async def evaluate_async(
        self,
        config: "EvaluatorServerConfig",
        experiment_id: str,
    ) -> None:
        self._config = config
        await self._evaluate_inner(
            cloudevent_unary_send=self.queue_cloudevent,
            output_bus=self.output_bus,
            experiment_id=experiment_id,
        )

    async def _evaluate_inner(  # pylint: disable=too-many-branches
        self,
        cloudevent_unary_send: Callable[
            [Union[CloudEvent, _ert_com_protocol.DispatcherMessage]], Awaitable[None]
        ],
        output_bus: Optional[
            "asyncio.Queue[_ert_com_protocol.DispatcherMessage]"
        ] = None,
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
        timeout_queue: "asyncio.Queue[Union[CloudEvent, _ert_com_protocol.DispatcherMessage]]" = (  # noqa: E501 # pylint: disable=line-too-long
            asyncio.Queue()
        )
        # Based on the experiment id the generator will
        # give a function returning cloud event or protobuf
        event_creator = self.generate_event_creator(experiment_id=experiment_id)
        on_timeout, send_timeout_future = self.setup_timeout_callback(
            timeout_queue, cloudevent_unary_send, event_creator
        )

        if not self.id_:
            raise ValueError("Ensemble id not set")
        if not self._config:
            raise ValueError("no config")  # mypy

        # event for normal evaluation, will be overwritten later in case of failure
        # or cancellation
        result = event_creator(identifiers.EVTYPE_ENSEMBLE_STOPPED, None)

        try:
            # Dispatch STARTED-event
            out_cloudevent = event_creator(identifiers.EVTYPE_ENSEMBLE_STARTED, None)
            await cloudevent_unary_send(out_cloudevent)

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
            sema = threading.BoundedSemaphore(value=CONCURRENT_INTERNALIZATION)
            if (
                output_bus and experiment_id is not None
            ):  # when running experiment server
                self._job_queue.add_dispatch_information_to_jobs_file(
                    ens_id=self.id_,
                    dispatch_url=self._config.dispatch_uri,
                    cert=self._config.cert,
                    token=self._config.token,
                    experiment_id=experiment_id,
                )
                # Finally, run the queue-loop until it finishes or raises
                await self._job_queue.execute_queue_comms_via_bus(
                    experiment_id=experiment_id,
                    ens_id=self.id_,
                    pool_sema=sema,
                    evaluators=queue_evaluators,  # type: ignore
                    output_bus=output_bus,
                )
            else:
                self._job_queue.add_dispatch_information_to_jobs_file(
                    ens_id=self.id_,
                    dispatch_url=self._config.dispatch_uri,
                    cert=self._config.cert,
                    token=self._config.token,
                )
                # Finally, run the queue-loop until it finishes or raises
                await self._job_queue.execute_queue_via_websockets(
                    self._config.dispatch_uri,
                    self.id_,
                    sema,
                    queue_evaluators,  # type: ignore
                    cert=self._config.cert,
                    token=self._config.token,
                )

        except asyncio.CancelledError:
            logger.debug("ensemble was cancelled")
            result = event_creator(identifiers.EVTYPE_ENSEMBLE_CANCELLED, None)

        except Exception:  # pylint: disable=broad-except
            logger.exception(
                "unexpected exception in ensemble",
                exc_info=True,
            )
            result = event_creator(identifiers.EVTYPE_ENSEMBLE_FAILED, None)

        else:
            logger.debug("ensemble finished normally")

        finally:
            await timeout_queue.put(None)  # signal to exit timer
            await send_timeout_future

            # Dispatch final result from evaluator - FAILED, CANCEL or STOPPED
            assert self._config  # mypy
            await cloudevent_unary_send(result)

    @property
    def cancellable(self) -> bool:
        return True

    def cancel(self) -> None:
        self._job_queue.kill_all_jobs()
        logger.debug("evaluator cancelled")

    @property
    def output_bus(
        self,
    ) -> "asyncio.Queue[_ert_com_protocol.DispatcherMessage]":
        return self._output_bus
