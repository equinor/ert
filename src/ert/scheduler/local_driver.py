from __future__ import annotations

import asyncio
import os
from asyncio.subprocess import Process
from typing import MutableMapping

from ert.scheduler.driver import Driver
from ert.scheduler.event import FinishedEvent, StartedEvent

_TERMINATE_TIMEOUT = 10.0


class LocalDriver(Driver):
    def __init__(self) -> None:
        super().__init__()
        self._tasks: MutableMapping[int, asyncio.Task[None]] = {}

    async def submit(
        self, iens: int, executable: str, /, *args: str, cwd: str, name: str = "dummy"
    ) -> None:
        await self.kill(iens)
        self._tasks[iens] = asyncio.create_task(
            self._run(iens, executable, *args, cwd=cwd)
        )

    async def kill(self, iens: int) -> None:
        try:
            self._tasks[iens].cancel()
            await self._tasks[iens]
            del self._tasks[iens]
        except (KeyError, asyncio.CancelledError):
            return

    async def finish(self) -> None:
        await asyncio.gather(*self._tasks.values())

    async def _run(self, iens: int, executable: str, /, *args: str, cwd: str) -> None:
        try:
            proc = await self._init(
                iens,
                executable,
                *args,
                cwd=cwd,
            )
        except FileNotFoundError:
            # /bin/sh uses returncode 127 for FileNotFound, so copy that
            # behaviour.
            await self.event_queue.put(FinishedEvent(iens=iens, returncode=127))
            return

        await self.event_queue.put(StartedEvent(iens=iens))
        try:
            returncode = await self._wait(proc)
            await self.event_queue.put(FinishedEvent(iens=iens, returncode=returncode))
        except asyncio.CancelledError:
            returncode = await self._kill(proc)
            await self.event_queue.put(
                FinishedEvent(iens=iens, returncode=returncode, aborted=True)
            )

    async def _init(
        self, iens: int, executable: str, /, *args: str, cwd: str
    ) -> Process:
        """This method exists to allow for mocking it in tests"""
        return await asyncio.create_subprocess_exec(
            executable,
            *args,
            cwd=cwd,
            preexec_fn=os.setpgrp,
        )

    async def _wait(self, proc: Process) -> int:
        """This method exists to allow for mocking it in tests"""
        return await proc.wait()

    async def _kill(self, proc: Process) -> int:
        """This method exists to allow for mocking it in tests"""
        try:
            proc.terminate()
            await asyncio.wait_for(proc.wait(), _TERMINATE_TIMEOUT)
        except asyncio.TimeoutError:
            proc.kill()
        return await proc.wait()

    async def poll(self) -> None:
        """LocalDriver does not poll"""
