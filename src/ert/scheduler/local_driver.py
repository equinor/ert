from __future__ import annotations

import asyncio
import contextlib
import logging
import multiprocessing
import signal
from contextlib import suppress
from pathlib import Path
from typing import MutableMapping, Optional, Set

from .driver import SIGNAL_OFFSET, Driver
from .event import FinishedEvent, StartedEvent

_TERMINATE_TIMEOUT = 10.0

logger = logging.getLogger(__name__)


class LocalDriver(Driver):
    def __init__(self) -> None:
        super().__init__()
        self._tasks: MutableMapping[int, asyncio.Task[None]] = {}
        self._sent_finished_events: Set[int] = set()
        self._spawn_context = multiprocessing.get_context("spawn")

    async def submit(
        self,
        iens: int,
        executable: str,
        /,
        *args: str,
        name: str = "dummy",
        runpath: Optional[Path] = None,
        num_cpu: Optional[int] = 1,
        realization_memory: Optional[int] = 0,
    ) -> None:
        self._tasks[iens] = asyncio.create_task(self._run(iens, executable, *args))
        with suppress(KeyError):
            self._sent_finished_events.remove(iens)

    async def kill(self, iens: int) -> None:
        try:
            self._tasks[iens].cancel()
            logger.info(f"Killing realization {iens}")
            with contextlib.suppress(asyncio.CancelledError):
                await self._tasks[iens]
            del self._tasks[iens]
            await self._dispatch_finished_event(iens, signal.SIGTERM + SIGNAL_OFFSET)

        except KeyError:
            logger.info(f"Realization {iens} is already killed")
            return
        except Exception as err:
            logger.error(f"Killing realization {iens} failed with error {err}")
            raise err

    async def finish(self) -> None:
        results = await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Exception in LocalDriver: {result}")
                raise result
        logger.info("All realization tasks finished")

    async def _run(self, iens: int, executable: str, /, *args: str) -> None:
        logger.debug(
            f"Submitting realization {iens} as command '{executable} {' '.join(args)}'"
        )
        try:
            proc = await self._init(
                iens,
                executable,
                *args,
            )
        except FileNotFoundError as err:
            # /bin/sh uses returncode 127 for FileNotFound, so copy that
            # behaviour.
            msg = f"Realization {iens} failed with {err}"
            logger.error(msg)
            self._job_error_message_by_iens[iens] = msg
            await self._dispatch_finished_event(iens, 127)
            return

        try:
            proc.start()
            await self.event_queue.put(StartedEvent(iens=iens))

            returncode = 1
            while proc.is_alive():
                proc.join(timeout=0.001)
                await asyncio.sleep(1)
            logger.info(f"Realization {iens} finished with exitcode={proc.exitcode}")
            returncode = proc.exitcode if proc.exitcode is not None else 1
        except asyncio.CancelledError:
            returncode = await self._kill(proc)
        finally:
            await self._dispatch_finished_event(iens, returncode)

    async def _dispatch_finished_event(self, iens: int, returncode: int) -> None:
        """Dispatch a finished event unless we have already done so for a given realization (iens)"""
        if iens not in self._sent_finished_events:
            await self.event_queue.put(FinishedEvent(iens=iens, returncode=returncode))
            self._sent_finished_events.add(iens)

    async def _init(
        self, iens: int, executable: str, /, *args: str
    ) -> multiprocessing.Process:
        """This method exists to allow for mocking it in tests"""
        from _ert.forward_model_runner.job_dispatch import main  # noqa

        return self._spawn_context.Process(
            target=main, args=[["job_dispatch.py", *args]]
        )  # type: ignore

    @classmethod
    async def _wait(
        cls,
        proc: multiprocessing.Process,
        max_wait: int = 10,
        blocked_wait: float = 0.001,
        sleep_interval: int = 1,
    ) -> None:
        wait = 0
        while wait < max_wait and proc.exitcode is None:
            proc.join(timeout=blocked_wait)
            await asyncio.sleep(sleep_interval)
            wait += blocked_wait + sleep_interval

    @classmethod
    async def _kill(cls, proc: multiprocessing.Process) -> int:
        proc.terminate()
        await cls._wait(proc)
        proc.kill()
        await cls._wait(proc)
        # the returncode of a subprocess will be the negative signal value
        # if it terminated due to a signal.
        # https://docs.python.org/3/library/subprocess.html#subprocess.CompletedProcess.returncode
        # we return SIGNAL_OFFSET + signal value to be in line with lfs/pbs drivers.
        if proc.exitcode is not None:
            return proc.exitcode + SIGNAL_OFFSET
        else:
            return -1 + SIGNAL_OFFSET

    async def poll(self) -> None:
        """LocalDriver does not poll"""
