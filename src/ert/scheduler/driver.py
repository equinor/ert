from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from enum import Enum
from typing import (
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

    def __init__(self) -> None:
        self.event_queue: Optional[asyncio.Queue[Tuple[int, JobEvent]]] = None

    async def ainit(self) -> None:
        if self.event_queue is None:
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
