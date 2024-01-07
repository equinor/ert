from __future__ import annotations

import asyncio
import os
from asyncio.subprocess import Process
from typing import MutableMapping

from ert.scheduler.driver import Driver, JobEvent

_TERMINATE_TIMEOUT = 10.0


class LocalDriver(Driver):
    def __init__(self) -> None:
        super().__init__()
        self._tasks: MutableMapping[int, asyncio.Task[None]] = {}

    async def submit(self, iens: int, executable: str, /, *args: str, cwd: str) -> None:
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
        except Exception as exc:
            print(f"{exc=}")
            await self.event_queue.put((iens, JobEvent.FAILED))
            return

        await self.event_queue.put((iens, JobEvent.STARTED))
        try:
            if await self._wait(proc):
                await self.event_queue.put((iens, JobEvent.COMPLETED))
            else:
                await self.event_queue.put((iens, JobEvent.FAILED))
        except asyncio.CancelledError:
            await self._kill(proc)
            await self.event_queue.put((iens, JobEvent.ABORTED))

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

    async def _wait(self, proc: Process) -> bool:
        """This method exists to allow for mocking it in tests"""
        return await proc.wait() == 0

    async def _kill(self, proc: Process) -> None:
        """This method exists to allow for mocking it in tests"""
        try:
            proc.terminate()
            await asyncio.wait_for(proc.wait(), _TERMINATE_TIMEOUT)
        except asyncio.TimeoutError:
            proc.kill()
            await asyncio.wait_for(proc.wait(), _TERMINATE_TIMEOUT)

    async def poll(self) -> None:
        """LocalDriver does not poll"""
