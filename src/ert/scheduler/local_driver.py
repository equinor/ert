from __future__ import annotations

import asyncio
import os
from typing import MutableMapping

from ert.scheduler.driver import Driver, JobEvent


class LocalDriver(Driver):
    def __init__(self) -> None:
        super().__init__()
        self._tasks: MutableMapping[int, asyncio.Task[None]] = {}

    async def submit(self, iens: int, executable: str, /, *args: str, cwd: str) -> None:
        self._tasks[iens] = asyncio.create_task(
            self._wait_until_finish(iens, executable, *args, cwd=cwd)
        )

    async def kill(self, iens: int) -> None:
        try:
            self._tasks[iens].cancel()
        except KeyError:
            return

    async def _wait_until_finish(
        self, iens: int, executable: str, /, *args: str, cwd: str
    ) -> None:
        proc = await asyncio.create_subprocess_exec(
            executable,
            *args,
            cwd=cwd,
            preexec_fn=os.setpgrp,
        )

        if self.event_queue is None:
            await self.ainit()
        assert self.event_queue is not None

        await self.event_queue.put((iens, JobEvent.STARTED))
        try:
            if await proc.wait() == 0:
                await self.event_queue.put((iens, JobEvent.COMPLETED))
            else:
                await self.event_queue.put((iens, JobEvent.FAILED))
        except asyncio.CancelledError:
            proc.terminate()
            await self.event_queue.put((iens, JobEvent.ABORTED))
