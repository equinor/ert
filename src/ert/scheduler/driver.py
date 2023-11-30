import asyncio
import logging
import os
import re
import shlex
from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

from ert.config.parsing.queue_system import QueueSystem

if TYPE_CHECKING:
    from ert.config import QueueConfig
    from ert.scheduler import RealizationState


logger = logging.getLogger(__name__)


class Driver(ABC):
    def __init__(
        self,
        options: Optional[Sequence[Tuple[str, str]]] = None,
    ):
        self.options: Dict[str, str] = dict(options or [])

    @abstractmethod
    async def submit(self, realization: "RealizationState") -> None:
        pass

    @abstractmethod
    async def poll_statuses(self) -> None:
        pass

    @abstractmethod
    async def kill(self, realization: "RealizationState") -> None:
        pass

    @classmethod
    def create_driver(cls, queue_config: "QueueConfig") -> "Driver":
        if queue_config.queue_system == QueueSystem.LOCAL:
            return LocalDriver(queue_config.queue_options.get(QueueSystem.LOCAL, []))
        elif queue_config.queue_system == QueueSystem.LSF:
            return LSFDriver(queue_config.queue_options.get(QueueSystem.LSF, []))
        raise NotImplementedError


class LocalDriver(Driver):
    def __init__(self, queue_config: List[Tuple[str, str]]):
        super().__init__(queue_config)

        self._processes: Dict["RealizationState", "asyncio.subprocess.Process"] = {}
        self._currently_polling = False

    @property
    def optionnames(self) -> List[str]:
        return []

    async def submit(self, realization: "RealizationState") -> None:
        """Submit and *actually (a)wait* for the process to finish."""
        realization.accept()
        try:
            process = await asyncio.create_subprocess_exec(
                realization.realization.job_script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=realization.realization.run_arg.runpath,
                preexec_fn=os.setpgrp,
            )
        except Exception as exc:
            print(exc)
            raise
        realization.start()
        print(
            f"Started realization {realization.realization.run_arg.iens} with pid {process.pid}"
        )
        self._processes[realization] = process

        # Wait for process to finish:
        output, error = await process.communicate()
        if process.returncode == 0:
            if output:
                logger.info(output)
            if str(realization.current_state.id) == "RUNNING":
                realization.runend()
            else:
                logger.debug(
                    f"Realization {realization.realization.run_arg.iens} finished "
                    f"successfully but was in state {realization.current_state.id}"
                )
        else:
            if output:
                logger.error(output)
            if error:
                logger.error(error)
            if str(realization.current_state.id) == "RUNNING":  # (circular import..)
                realization.runfail()

    async def poll_statuses(self) -> None:
        pass

    async def kill(self, realization: "RealizationState") -> None:
        self._processes[realization].kill()
        realization.verify_kill()


class LSFDriver(Driver):
    LSF_STATUSES = [
        "PEND",
        "SSUSP",
        "PSUSP",
        "USUSP",
        "RUN",
        "EXIT",
        "ZOMBI",
        "DONE",
        "PDONE",
        "UNKWN",
    ]

    def __init__(self, queue_options: Optional[List[Tuple[str, str]]]):
        super().__init__(queue_options)

        self._realstate_to_lsfid: Dict["RealizationState", str] = {}
        self._lsfid_to_realstate: Dict[str, "RealizationState"] = {}
        self._max_attempt: int = 100
        self._submit_processes: Dict[
            "RealizationState", "asyncio.subprocess.Process"
        ] = {}
        self._retry_sleep_period = 3
        self._currently_polling = False

    async def run_with_retries(
        self, func: Callable[[], Awaitable[Any]], error_msg: str = ""
    ) -> None:
        current_attempt = 0
        while current_attempt < self._max_attempt:
            current_attempt += 1
            try:
                function_output = await func()
                if function_output:
                    return function_output
                await asyncio.sleep(self._retry_sleep_period)
            except asyncio.CancelledError as e:
                logger.error(e)
                await asyncio.sleep(self._retry_sleep_period)
        raise RuntimeError(error_msg)

    async def submit(self, realization: "RealizationState") -> None:
        submit_cmd = self.build_submit_cmd(
            "-J",
            f"poly_{realization.realization.run_arg.iens}",
            str(realization.realization.job_script),
            str(realization.realization.run_arg.runpath),
        )
        await self.run_with_retries(
            lambda: self._submit(submit_cmd, realization=realization),
            error_msg="Maximum number of submit errors exceeded\n",
        )

    async def _submit(
        self, submit_command: List[str], realization: "RealizationState"
    ) -> bool:
        result = await self.run_shell_command(submit_command, command_name="bsub")
        if not result:
            return False

        (process, output, error) = result
        self._submit_processes[realization] = process
        lsf_id_match = re.match(
            "Job <\\d+> is submitted to \\w+ queue <\\w+>\\.", output.decode()
        )
        if lsf_id_match is None:
            logger.error(f"Could not parse lsf id from: {output.decode()}")
            return False
        lsf_id = lsf_id_match.group(0)
        self._realstate_to_lsfid[realization] = lsf_id
        self._lsfid_to_realstate[lsf_id] = realization
        realization.accept()
        logger.info(f"Submitted job {realization} and got LSF JOBID {lsf_id}")
        return True

    def build_submit_cmd(self, *args: str) -> List[str]:
        submit_cmd = [self.options.get("BSUB_CMD", "bsub")]
        if (lsf_queue := self.options.get("LSF_QUEUE")) is not None:
            submit_cmd += ["-q", lsf_queue]

        return [*submit_cmd, *args]

    async def run_shell_command(
        self, command_to_run: List[str], command_name: str = ""
    ) -> Optional[Tuple[asyncio.subprocess.Process, bytes, bytes]]:
        process = await asyncio.create_subprocess_exec(
            *command_to_run,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        output, _error = await process.communicate()
        if process.returncode != 0:
            logger.error(
                (
                    f"{command_name} returned non-zero exitcode: {process.returncode}\n"
                    f"{output.decode()}\n"
                    f"{_error.decode()}"
                )
            )
            return None
        return (process, output, _error)

    async def poll_statuses(self) -> None:
        if self._currently_polling:
            logger.debug("Already polling status elsewhere")
            return

        if not self._realstate_to_lsfid:
            # Nothing has been submitted yet.
            logger.warning("Skipped polling due to no jobs submitted")
            return

        poll_cmd = [
            self.options.get("BJOBS_CMD", "bjobs"),
            *self._realstate_to_lsfid.values(),
        ]
        try:
            await self.run_with_retries(lambda: self._poll_statuses(poll_cmd))
        # suppress runtime error
        except RuntimeError:
            return
        except ValueError as e:
            # raise this value error as runtime error
            raise RuntimeError(e)

    async def _poll_statuses(self, poll_cmd: List[str]) -> bool:
        self._currently_polling = True
        result = await self.run_shell_command(poll_cmd, command_name="bjobs")

        if result is None:
            return False
        (_, output, _) = result

        for line in output.decode(encoding="utf-8").split("\n"):
            if "JOBID" in line:
                continue
            tokens = shlex.split(
                line
            )  # (doing shlex parsing is actually wrong, positions are fixed by byte positions)
            if not tokens:
                continue
            if tokens[0] not in self._lsfid_to_realstate:
                # A LSF id we know nothing of, this should not happen.
                raise ValueError(f"Found unknown job id ({tokens[0]})")

            realstate = self._lsfid_to_realstate[tokens[0]]

            if tokens[2] == "PEND" and str(realstate.current_state.id) == "WAITING":
                # we want RealizationState.RUNNING but circular import
                realstate.accept()
            if tokens[2] == "RUN" and str(realstate.current_state.id) == "WAITING":
                realstate.accept()
                realstate.start()
            if tokens[2] == "RUN" and str(realstate.current_state.id) == "PENDING":
                realstate.start()
            if tokens[2] == "DONE" and str(realstate.current_state.id) == "WAITING":
                # This warrants something smarter, that will allow us to
                # automatically go through the states up until DONE..
                realstate.accept()
                realstate.start()
                realstate.runend()
            if tokens[2] == "DONE" and str(realstate.current_state.id) == "PENDING":
                realstate.start()
                realstate.runend()
            if tokens[2] == "DONE" and str(realstate.current_state.id) == "RUNNING":
                realstate.runend()
            if tokens[2] not in LSFDriver.LSF_STATUSES:
                raise ValueError(
                    f"The lsf_status {tokens[2]} for job {tokens[0]} was not recognized\n"
                )

        self._currently_polling = False
        return True

    async def kill(self, realization: "RealizationState") -> None:
        lsf_job_id = self._realstate_to_lsfid[realization]
        logger.debug(f"Attempting to kill {lsf_job_id=}")
        kill_cmd = [
            self.options.get("BKILL_CMD", "bkill"),
            lsf_job_id,
        ]
        await self.run_with_retries(
            lambda: self._kill(kill_cmd, realization, lsf_job_id),
            error_msg="Maximum number of kill errors exceeded\n",
        )

    async def _kill(
        self, kill_cmd: List[str], realization: "RealizationState", lsf_job_id: str
    ) -> bool:
        result = await self.run_shell_command(kill_cmd, "bkill")
        if result is None:
            return False
        realization.verify_kill()
        logger.info(f"Successfully killed job {lsf_job_id}")
        return True
