from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
from asyncio.subprocess import Process
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

        await self.event_queue.put(StartedEvent(iens=iens))

        returncode = 1
        try:
            returncode = await self._wait(proc)
            logger.info(f"Realization {iens} finished with {returncode=}")
        except asyncio.CancelledError:
            returncode = await self._kill(proc)
        finally:
            await self._dispatch_finished_event(iens, returncode)

    async def _dispatch_finished_event(self, iens: int, returncode: int) -> None:
        """Dispatch a finished event unless we have already done so for a given realization (iens)"""
        if iens not in self._sent_finished_events:
            await self.event_queue.put(FinishedEvent(iens=iens, returncode=returncode))
            self._sent_finished_events.add(iens)

    @staticmethod
    async def _init(iens: int, executable: str, /, *args: str) -> Process:
        """This method exists to allow for mocking it in tests"""
        return await asyncio.create_subprocess_exec(
            executable,
            *args,
            preexec_fn=os.setpgrp,
        )

    @staticmethod
    async def _wait(proc: Process) -> int:
        return await proc.wait()

    @staticmethod
    async def _kill(proc: Process) -> int:
        try:
            proc.terminate()
            await asyncio.wait_for(proc.wait(), _TERMINATE_TIMEOUT)
        except asyncio.TimeoutError:
            proc.kill()
        except ProcessLookupError:
            # This will happen if the subprocess has not yet started
            return signal.SIGTERM + SIGNAL_OFFSET
        ret_val = await proc.wait()
        # the returncode of a subprocess will be the negative signal value
        # if it terminated due to a signal.
        # https://docs.python.org/3/library/subprocess.html#subprocess.CompletedProcess.returncode
        # we return SIGNAL_OFFSET + signal value to be in line with lfs/pbs drivers.
        return -ret_val + SIGNAL_OFFSET

    async def poll(self) -> None:
        """LocalDriver does not poll"""
