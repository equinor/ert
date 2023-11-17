import asyncio
import shlex
import shutil
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from ert.config import QueueConfig, QueueSystem

if TYPE_CHECKING:
    from ert.job_queue import QueueableRealization, RealizationState


class Driver(ABC):
    def __init__(
        self,
        options: Optional[List[Tuple[str, str]]] = None,
    ):
        self._options = {}

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
    async def submit(self, realization: "QueueableRealization"):
        pass

    @abstractmethod
    async def poll_statuses(self):
        pass

    @abstractmethod
    async def kill(self, realization: "QueueableRealization"):
        pass

    @classmethod
    def create_driver(cls, queue_config: QueueConfig) -> "Driver":
        if queue_config.queue_system == QueueSystem.LOCAL:
            return LocalDriver(queue_config.queue_options[QueueSystem.LOCAL])
        elif queue_config.queue_system == QueueSystem.LSF:
            return LSFDriver(queue_config.queue_options[QueueSystem.LSF])
        raise NotImplementedError


class LocalDriver(Driver):
    def __init__(self, queue_config: List[Tuple[str, str]]):
        super().__init__(queue_config)

        self._processes: Dict["RealizationState", "asyncio.subprocess.Process"] = {}
        self._currently_polling = False

    async def submit(self, realization: "RealizationState"):
        """Submit and *actually (a)wait* for the process to finish."""
        realization.accept()
        try:
            process = await asyncio.create_subprocess_exec(
                realization.realization.job_script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=realization.realization.run_arg.runpath,
            )
        except Exception as exc:
            print(exc)
            raise
        if process.returncode is None:
            realization.start()
        else:
            # Hmm, can it return so fast that we have a zero return code here?
            raise RuntimeError
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
                # (we might be killed)
                realization.runfail()
            # TODO: fetch stdout/stderr

    async def poll_statuses(self):
        pass

    async def kill(self, realization: "RealizationState"):
        self._processes[realization].kill()
        realization.verify_kill()


class LSFDriver(Driver):
    def __init__(self, queue_options):
        super().__init__(queue_options)

        self._realstate_to_lsfid: Dict["RealizationState", str] = {}
        self._lsfid_to_realstate: Dict[str, "RealizationState"] = {}
        self._submit_processes: Dict[
            "RealizationState", "asyncio.subprocess.Process"
        ] = {}

        self._currently_polling = False

    async def submit(self, realization: "RealizationState"):
        submit_cmd = [
            "bsub",
            "-J",
            f"poly_{realization.realization.run_arg.iens}",
            realization.realization.job_script,
            realization.realization.run_arg.runpath,
        ]
        assert shutil.which(submit_cmd[0])  # does not propagate back..
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
            print(f"ERROR: Could not parse lsf id from: {output}")

    async def poll_statuses(self) -> None:
        if self._currently_polling:
            # Don't repeat if we are called too often.
            # So easy in async..
            return self._statuses
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

    async def kill(self, realization):
        print(f"would like to kill {realization}")
        pass
