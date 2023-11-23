import asyncio
import os
import shlex
import shutil
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from ert.config.parsing.queue_system import QueueSystem

if TYPE_CHECKING:
    from ert.config import QueueConfig
    from ert.job_queue import RealizationState


class Driver(ABC):
    def __init__(
        self,
        options: Optional[List[Tuple[str, str]]] = None,
    ):
        self._options: Dict[str, str] = {}

        if options:
            for key, value in options:
                self.set_option(key, value)

    def set_option(self, option: str, value: str) -> None:
        self._options.update({option: value})

    def get_option(self, option_key: str) -> str:
        return self._options[option_key]

    def has_option(self, option_key: str) -> bool:
        return option_key in self._options

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
        print(output)
        print(error)
        if process.returncode == 0:
            realization.runend()
        else:
            if str(realization.current_state.id) == "RUNNING":  # (circular import..)
                realization.runfail()
            # TODO: fetch stdout/stderr

    async def poll_statuses(self) -> None:
        pass

    async def kill(self, realization: "RealizationState") -> None:
        self._processes[realization].kill()
        realization.verify_kill()


class LSFDriver(Driver):
    def __init__(self, queue_options: Optional[List[Tuple[str, str]]]):
        super().__init__(queue_options)

        self._realstate_to_lsfid: Dict["RealizationState", str] = {}
        self._lsfid_to_realstate: Dict[str, "RealizationState"] = {}
        self._submit_processes: Dict[
            "RealizationState", "asyncio.subprocess.Process"
        ] = {}

        self._currently_polling = False

    async def submit(self, realization: "RealizationState") -> None:
        submit_cmd: List[str] = [
            "bsub",
            "-J",
            f"poly_{realization.realization.run_arg.iens}",
            str(realization.realization.job_script),
            str(realization.realization.run_arg.runpath),
        ]
        process = await asyncio.create_subprocess_exec(
            *submit_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._submit_processes[realization] = process

        # Wait for submit process to finish:
        output, error = await process.communicate()
        print(output)  # FLAKY ALERT, we seem to get empty
        print(error)

        try:
            lsf_id = str(output).split(" ")[1].replace("<", "").replace(">", "")
            self._realstate_to_lsfid[realization] = lsf_id
            self._lsfid_to_realstate[lsf_id] = realization
            realization.accept()
            print(f"Submitted job {realization} and got LSF JOBID {lsf_id}")
        except Exception:
            # We should probably retry the submission, bsub stdout seems flaky.
            print(f"ERROR: Could not parse lsf id from: {output!r}")

    async def poll_statuses(self) -> None:
        if self._currently_polling:
            return
        self._currently_polling = True

        if not self._realstate_to_lsfid:
            # Nothing has been submitted yet.
            return

        poll_cmd = ["bjobs"] + list(self._realstate_to_lsfid.values())
        assert shutil.which(poll_cmd[0])  # does not propagate back..
        process = await asyncio.create_subprocess_exec(
            *poll_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        output, _error = await process.communicate()
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
                continue
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

        self._currently_polling = False

    async def kill(self, realization: "RealizationState") -> None:
        print(f"would like to kill {realization}")
        pass
