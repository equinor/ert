from __future__ import annotations

import asyncio
import logging
import shlex
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from ert.scheduler.event import Event

SIGNAL_OFFSET = 128
"""Bash and other shells add an offset of 128 to the signal value when a process exited due to a signal"""


class Driver(ABC):
    """Adapter for the HPC cluster."""

    def __init__(self, **kwargs: Dict[str, str]) -> None:
        self._event_queue: Optional[asyncio.Queue[Event]] = None

    @property
    def event_queue(self) -> asyncio.Queue[Event]:
        if self._event_queue is None:
            self._event_queue = asyncio.Queue()
        return self._event_queue

    @abstractmethod
    async def submit(
        self,
        iens: int,
        executable: str,
        /,
        *args: str,
        name: str = "dummy",
        runpath: Optional[Path] = None,
    ) -> None:
        """Submit a program to execute on the cluster.

        Args:
          iens: Realization number. (Unique for each job)
          executable: Program to execute.
          args: List of arguments to send to the program.
          cwd: Working directory.
          name: Name of job as submitted to compute cluster
        """

    @abstractmethod
    async def kill(self, iens: int) -> None:
        """Terminate execution of a job associated with a realization.

        Args:
          iens: Realization number.
        """

    @abstractmethod
    async def poll(self) -> None:
        """Poll for new job events"""

    @abstractmethod
    async def finish(self) -> None:
        """make sure that all the jobs / realizations are complete."""

    async def _execute_with_retry(
        self,
        cmd_with_args: List[str],
        retry_codes: Iterable[int] = (),
        accept_codes: Iterable[int] = (),
        stdin: Optional[bytes] = None,
        retries: int = 1,
        retry_interval: float = 1.0,
        driverlogger: Optional[logging.Logger] = None,
    ) -> Tuple[bool, str]:
        _logger = driverlogger or logging.getLogger(__name__)
        error_message: Optional[str] = None

        for _ in range(retries):
            process = await asyncio.create_subprocess_exec(
                *cmd_with_args,
                stdin=asyncio.subprocess.PIPE if stdin else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate(stdin)

            assert process.returncode is not None
            if process.returncode == 0:
                return True, stdout.decode(errors="ignore").strip()
            elif process.returncode in retry_codes:
                error_message = stderr.decode(errors="ignore").strip()
            elif process.returncode in accept_codes:
                return True, stderr.decode(errors="ignore").strip()
            else:
                error_message = (
                    f'Command "{shlex.join(cmd_with_args)}" failed '
                    f"with exit code {process.returncode} and error message: "
                    + stderr.decode(errors="ignore").strip()
                )
                _logger.error(error_message)
                return False, error_message

            await asyncio.sleep(retry_interval)
        error_message = (
            f'Command "{shlex.join(cmd_with_args)}" failed after {retries} retries'
            f" with error {error_message or '<empty>'}"
        )
        _logger.error(error_message)
        return False, error_message
