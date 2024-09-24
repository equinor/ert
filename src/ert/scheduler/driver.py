from __future__ import annotations

import asyncio
import logging
import shlex
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .event import Event

SIGNAL_OFFSET = 128
"""Bash and other shells add an offset of 128 to the signal value when a process exited due to a signal"""


class Driver(ABC):
    """Adapter for the HPC cluster."""

    def __init__(self, **kwargs: Dict[str, str]) -> None:
        self._event_queue: Optional[asyncio.Queue[Event]] = None
        self._job_error_message_by_iens: Dict[int, str] = {}

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
        num_cpu: Optional[int] = 1,
        realization_memory: Optional[int] = 0,
    ) -> None:
        """Submit a program to execute on the cluster.

        Args:
          iens: Realization number. (Unique for each job)
          executable: Program to execute.
          args: List of arguments to send to the program.
          cwd: Working directory.
          name: Name of job as submitted to compute cluster
          num_cpu: Number of CPU-cores to allocate
          realization_memory: Memory to book, in bytes. 0 means no booking. This should
            be regareded as a hint to the queue system, not absolute limits.
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

    def read_stdout_and_stderr_files(
        self, runpath: str, job_name: str, num_characters_to_read_from_end: int = 300
    ) -> str:
        """Each driver should provide some output in case of failure."""
        return ""

    @staticmethod
    async def _execute_with_retry(
        cmd_with_args: List[str],
        retry_on_empty_stdout: Optional[bool] = False,
        retry_codes: Iterable[int] = (),
        accept_codes: Iterable[int] = (),
        stdin: Optional[bytes] = None,
        total_attempts: int = 1,
        retry_interval: float = 1.0,
        driverlogger: Optional[logging.Logger] = None,
        return_on_msgs: Iterable[str] = (),
        error_on_msgs: Iterable[str] = (),
        log_to_debug: Optional[bool] = True,
    ) -> Tuple[bool, str]:
        _logger = driverlogger or logging.getLogger(__name__)
        error_message: Optional[str] = None

        for _ in range(total_attempts):
            try:
                process = await asyncio.create_subprocess_exec(
                    *cmd_with_args,
                    stdin=asyncio.subprocess.PIPE if stdin else None,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            except FileNotFoundError as e:
                return (False, str(e))

            stdout, stderr = await process.communicate(stdin)

            assert process.returncode is not None
            outputs = (
                f"exit code {process.returncode}, "
                f'output: "{stdout.decode(errors="ignore").strip() or "<empty>"}", and '
                f'error: "{stderr.decode(errors="ignore").strip() or "<empty>"}"'
            )
            if process.returncode == 0:
                if retry_on_empty_stdout and not stdout:
                    _logger.warning(
                        f'Command "{shlex.join(cmd_with_args)}" gave exit code 0 but empty stdout, '
                        "will retry. "
                        f'stderr: "{stderr.decode(errors="ignore").strip() or "<empty>"}"'
                    )
                else:
                    if log_to_debug:
                        _logger.debug(
                            f'Command "{shlex.join(cmd_with_args)}" succeeded with {outputs}'
                        )
                    return True, stdout.decode(errors="ignore").strip()
            elif return_on_msgs and any(
                return_on_msg in stderr.decode(errors="ignore")
                for return_on_msg in return_on_msgs
            ):
                return True, stderr.decode(errors="ignore").strip()
            elif error_on_msgs and any(
                error_on_msg in stderr.decode(errors="ignore")
                for error_on_msg in error_on_msgs
            ):
                return False, stderr.decode(errors="ignore").strip()
            elif process.returncode in retry_codes:
                error_message = outputs
            elif process.returncode in accept_codes:
                if log_to_debug:
                    _logger.debug(
                        f'Command "{shlex.join(cmd_with_args)}" succeeded with {outputs}'
                    )
                return True, stderr.decode(errors="ignore").strip()
            else:
                error_message = (
                    f'Command "{shlex.join(cmd_with_args)}" failed with {outputs}'
                )
                _logger.error(error_message)
                return False, error_message

            await asyncio.sleep(retry_interval)
        error_message = (
            f'Command "{shlex.join(cmd_with_args)}" failed after {total_attempts} attempts '
            f"with {outputs}"
        )
        _logger.error(error_message)
        return False, error_message
