from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Final, Union

from _ert import events
from _ert.events import (
    ForwardModelStepChecksum,
    ForwardModelStepFailure,
    ForwardModelStepRunning,
    ForwardModelStepStart,
    ForwardModelStepSuccess,
    event_to_json,
)
from _ert.forward_model_runner.client import (
    Client,
    ClientConnectionClosedOK,
    ClientConnectionError,
)
from _ert.forward_model_runner.reporting.base import Reporter
from _ert.forward_model_runner.reporting.message import (
    _JOB_EXIT_FAILED_STRING,
    Checksum,
    Exited,
    Finish,
    Init,
    Message,
    Running,
    Start,
)

logger = logging.getLogger(__name__)


class EventSentinel:
    pass


class Event(Reporter):
    """
    The Event reporter forwards events, coming from the running job, added with
    "report" to the given connection information.

    An Init event must be provided as the first message, which starts reporting,
    and a Finish event will signal the reporter that the last event has been reported.

    If event fails to be sent (e.g. due to connection error) it does not proceed to the
    next event but instead tries to re-send the same event.

    Whenever the Finish event (when all the jobs have exited) is provided
    the reporter will try to send all remaining events for a maximum of 60 seconds
    before stopping the reporter. Any remaining events will not be sent.
    """

    _sentinel: Final = EventSentinel()

    def __init__(self, evaluator_url, token=None, cert_path=None):
        self._evaluator_url = evaluator_url
        self._token = token
        if cert_path is not None:
            with open(cert_path, encoding="utf-8") as f:
                self._cert = f.read()
        else:
            self._cert = None

        self._ens_id = None
        self._real_id = None
        self._event_queue: asyncio.Queue[events.Event | EventSentinel] = asyncio.Queue()

        # seconds to timeout the reporter the thread after Finish() was received
        self._timeout_timestamp = None
        self._reporter_timeout = 60

        self._queue_polling_timeout = 2
        self._event_publishing_task = asyncio.create_task(self.async_event_publisher())
        self._event_publisher_ready = asyncio.Event()

    async def join(self) -> None:
        print("called join")
        await self._event_publishing_task

    async def stop(self) -> None:
        print("called stop")
        await self._event_queue.put(Event._sentinel)
        await self.join()

    async def async_event_publisher(self):
        logger.debug("Publishing event.")
        async with Client(
            url=self._evaluator_url,
            token=self._token,
            cert=self._cert,
        ) as client:
            self._event_publisher_ready.set()
            event = None
            while (
                self._timeout_timestamp is None
                or datetime.now() <= self._timeout_timestamp
            ):
                if event is None:
                    # if we successfully sent the event we can proceed
                    # to next one
                    try:
                        event = await asyncio.wait_for(
                            self._event_queue.get(), timeout=self._queue_polling_timeout
                        )
                    except asyncio.TimeoutError:
                        continue
                    if event is self._sentinel:
                        self._event_queue.task_done()
                        break
                try:
                    await client.send(event_to_json(event))
                    self._event_queue.task_done()
                    event = None
                    print("Sent event :)")
                except ClientConnectionError as exception:
                    # Possible intermittent failure, we retry sending the event
                    logger.error(str(exception))
                except ClientConnectionClosedOK as exception:
                    # The receiving end has closed the connection, we stop
                    # sending events
                    logger.debug(str(exception))
                    self._event_queue.task_done()
                    break
        print("TIMED OUT")

    async def report(self, msg: Message):
        await self._event_publisher_ready.wait()
        await self._report(msg)

    async def _report(self, msg: Message):
        if isinstance(msg, Init):
            await self._init_handler(msg)
        elif isinstance(msg, (Start, Running, Exited)):
            await self._job_handler(msg)
        elif isinstance(msg, Checksum):
            await self._checksum_handler(msg)
        elif isinstance(msg, Finish):
            await self._finished_handler()

    async def _dump_event(self, event: events.Event):
        print(f"DUMPED EVENT {type(event)=}")
        logger.debug(f'Schedule "{type(event)}" for delivery')
        await self._event_queue.put(event)

    async def _init_handler(self, msg: Init):
        self._ens_id = str(msg.ens_id)
        self._real_id = str(msg.real_id)

    async def _job_handler(self, msg: Union[Start, Running, Exited]):
        assert msg.job
        job_name = msg.job.name()
        job_msg = {
            "ensemble": self._ens_id,
            "real": self._real_id,
            "fm_step": str(msg.job.index),
        }
        if isinstance(msg, Start):
            logger.debug(f"Job {job_name} was successfully started")
            event = ForwardModelStepStart(
                **job_msg,
                std_out=str(Path(msg.job.std_out).resolve()),
                std_err=str(Path(msg.job.std_err).resolve()),
            )
            await self._dump_event(event)
            if not msg.success():
                logger.error(f"Job {job_name} FAILED to start")
                event = ForwardModelStepFailure(**job_msg, error_msg=msg.error_message)
                await self._dump_event(event)

        elif isinstance(msg, Exited):
            if msg.success():
                logger.debug(f"Job {job_name} exited successfully")
                await self._dump_event(ForwardModelStepSuccess(**job_msg))
            else:
                logger.error(
                    _JOB_EXIT_FAILED_STRING.format(
                        job_name=msg.job.name(),
                        exit_code=msg.exit_code,
                        error_message=msg.error_message,
                    )
                )
                event = ForwardModelStepFailure(
                    **job_msg, exit_code=msg.exit_code, error_msg=msg.error_message
                )
                await self._dump_event(event)

        elif isinstance(msg, Running):
            logger.debug(f"{job_name} job is running")
            event = ForwardModelStepRunning(
                **job_msg,
                max_memory_usage=msg.memory_status.max_rss,
                current_memory_usage=msg.memory_status.rss,
                cpu_seconds=msg.memory_status.cpu_seconds,
            )
            await self._dump_event(event)

    async def _finished_handler(self):
        await self._event_queue.put(Event._sentinel)
        self._timeout_timestamp = datetime.now() + timedelta(
            seconds=self._reporter_timeout
        )

    async def _checksum_handler(self, msg: Checksum):
        fm_checksum = ForwardModelStepChecksum(
            ensemble=self._ens_id,
            real=self._real_id,
            checksums={msg.run_path: msg.data},
        )
        await self._dump_event(fm_checksum)

    def cancel(self):
        self._event_publishing_task.cancel()
