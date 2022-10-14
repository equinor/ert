import asyncio
import contextlib
import os
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterator, Optional

import prefect

from _ert_job_runner.client import Client
from ert.async_utils import get_event_loop
from ert.data import FileTransformation
from ert.ensemble_evaluator.identifiers import (
    ERROR_MSG,
    EVTYPE_FM_JOB_FAILURE,
    EVTYPE_FM_JOB_START,
    EVTYPE_FM_JOB_SUCCESS,
    EVTYPE_FM_STEP_RUNNING,
    EVTYPE_FM_STEP_SUCCESS,
)

from ._io_map import _stage_transmitter_mapping
from ._job import UnixJob

if TYPE_CHECKING:
    import ert

    from ._step import Step

_BIN_FOLDER = "bin"


def _send_event(type_: str, source: str, data: Optional[Dict[str, Any]] = None) -> None:
    with Client(
        prefect.context.url,  # type: ignore  # pylint: disable=no-member
        prefect.context.token,  # type: ignore  # pylint: disable=no-member
        prefect.context.cert,  # type: ignore  # pylint: disable=no-member
    ) as client:
        client.send_event(
            ev_type=type_,
            ev_source=source,
            ev_data=data,
        )


@contextlib.contextmanager
def create_runpath(path: Optional[Path] = None) -> Iterator[Path]:
    if path is not None:
        yield Path(path)
    else:
        with tempfile.TemporaryDirectory() as run_path:
            yield Path(run_path)


class UnixTask(prefect.Task):  # type: ignore
    def __init__(
        self,
        step: "Step",
        output_transmitters: _stage_transmitter_mapping,
        ens_id: str,
        *args: Any,
        run_path: Optional[Path] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.step = step
        self._output_transmitters = output_transmitters
        self._ens_id = ens_id
        self._run_path = run_path

    def run_job(self, job: UnixJob, run_path: Path) -> None:
        shell_cmd = [job.executable.as_posix()]
        if job.args:
            shell_cmd += [os.path.expandvars(arg) for arg in job.args]
        env = os.environ.copy()
        env.update(
            {"PATH": (run_path / _BIN_FOLDER).as_posix() + ":" + os.environ["PATH"]}
        )
        cmd_exec = subprocess.run(
            shell_cmd,
            universal_newlines=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=run_path.as_posix(),
            env=env,
            check=False,
        )
        self.logger.info(cmd_exec.stderr)
        self.logger.info(cmd_exec.stdout)

        if cmd_exec.returncode != 0:
            self.logger.error(cmd_exec.stderr)
            _send_event(
                EVTYPE_FM_JOB_FAILURE,
                job.source(),
                {ERROR_MSG: cmd_exec.stderr},
            )
            raise OSError(
                f"Script {job.name} failed with exception {cmd_exec.stderr}\nOutput: "
                + f"{cmd_exec.stdout}"
            )

    def run_jobs(self, run_path: Path) -> None:
        for job in self.step.jobs:
            self.logger.info(f"Running command {job.name}")
            _send_event(
                EVTYPE_FM_JOB_START,
                job.source(),
            )
            if not isinstance(job, UnixJob):
                raise TypeError(f"unexpected job {type(job)} in unix task")
            self.run_job(job, run_path)
            _send_event(
                EVTYPE_FM_JOB_SUCCESS,
                job.source(),
            )

    def _load_and_dump_input(
        self,
        transmitters: _stage_transmitter_mapping,
        runpath: Path,
    ) -> None:
        async def transform_input(
            transformation: "ert.data.RecordTransformation",
            transmitter: "ert.data.RecordTransmitter",
        ) -> None:
            record = await transmitter.load()
            await transformation.from_record(
                record=record,
                root_path=runpath,
            )

        futures = []
        for input_ in self.step.inputs:
            if not input_.transformation:
                raise ValueError(f"no transformation for input '{input_.name}'")
            transmitter = transmitters[input_.name]
            futures.append(transform_input(input_.transformation, transmitter))
        get_event_loop().run_until_complete(asyncio.gather(*futures))

    def run(  # type: ignore  # pylint: disable=arguments-differ
        self,
        inputs: Optional[_stage_transmitter_mapping] = None,
    ) -> _stage_transmitter_mapping:
        async def transform_output(
            transformation: "ert.data.RecordTransformation",
            transmitter: "ert.data.RecordTransmitter",
            run_path: Path,
        ) -> None:
            record = await transformation.to_record(
                root_path=run_path,
            )
            await transmitter.transmit_record(record)

        with create_runpath(self._run_path) as run_path:
            if inputs is not None:
                self._load_and_dump_input(transmitters=inputs, runpath=run_path)
            _send_event(
                EVTYPE_FM_STEP_RUNNING,
                self.step.source(),
            )

            outputs: _stage_transmitter_mapping = {}
            self.run_jobs(run_path)

            futures = []
            for output in self.step.outputs:
                transformation = output.transformation
                if not transformation:
                    raise ValueError(f"no transformation for output '{output.name}'")
                if not isinstance(transformation, FileTransformation):
                    raise ValueError(
                        f"got unexpected transformation {transformation} for "
                        + f"'{output.name}'"
                    )
                if not (run_path / transformation.location).exists():
                    raise FileNotFoundError(
                        f"Output '{output.name}' file {transformation.location} was not"
                        + " generated!"
                    )

                transmitter = self._output_transmitters[output.name]
                outputs[output.name] = transmitter
                futures.append(transform_output(transformation, transmitter, run_path))

            get_event_loop().run_until_complete(asyncio.gather(*futures))
            _send_event(
                EVTYPE_FM_STEP_SUCCESS,
                self.step.source(),
            )
        return outputs
