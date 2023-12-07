from __future__ import annotations

import asyncio
import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import (
    MutableMapping,
    Optional,
    Tuple,
)


class JobEvent(Enum):
    STARTED = 0
    COMPLETED = 1
    FAILED = 2
    ABORTED = 3


class Driver(ABC):
    """Adapter for the HPC cluster."""

    event_queue: asyncio.Queue[Tuple[int, JobEvent]]

    def __init__(self) -> None:
        self.event_queue = asyncio.Queue()

    @abstractmethod
    async def submit(self, iens: int, executable: str, /, *args: str, cwd: str) -> None:
        """Submit a program to execute on the cluster.

        Args:
          iens: Realization number. (Unique for each job)
          executable: Program to execute.
          args: List of arguments to send to the program.
          cwd: Working directory.
        """

    @abstractmethod
    async def kill(self, iens: int) -> None:
        """Terminate execution of a job associated with a realization.

        Args:
          iens: Realization number.
        """

    def create_poll_task(self) -> Optional[asyncio.Task[None]]:
        """Create a `asyncio.Task` for polling the cluster.

        Returns:
          `asyncio.Task`, or None if polling is not applicable (eg. for LocalDriver)
        """

        return None


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
        await self.event_queue.put((iens, JobEvent.STARTED))
        try:
            if await proc.wait() == 0:
                await self.event_queue.put((iens, JobEvent.COMPLETED))
            else:
                await self.event_queue.put((iens, JobEvent.FAILED))
        except asyncio.CancelledError:
            proc.terminate()
            await self.event_queue.put((iens, JobEvent.ABORTED))
