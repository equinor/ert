import asyncio
from typing import Any, Dict, Optional
from pathlib import Path
import subprocess
import tempfile
import os

import prefect

from ert_shared.async_utils import get_event_loop
from ert_shared.ensemble_evaluator.client import Client
from ert_shared.ensemble_evaluator.entity import identifiers as ids

from ert.data import RecordTransmitter

_BIN_FOLDER = "bin"


def _send_event(type_: str, source: str, data: Optional[Dict[str, Any]] = None) -> None:
    with Client(
        prefect.context.url, prefect.context.token, prefect.context.cert
    ) as client:
        client.send_event(
            ev_type=type_,
            ev_source=source,
            ev_data=data,
        )


class UnixTask(prefect.Task):
    def __init__(self, step, output_transmitters, ee_id, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._step = step
        self._output_transmitters = output_transmitters
        self._ee_id = ee_id

    def get_step(self):
        return self._step

    def run_job(self, job: Any, run_path: Path):
        shell_cmd = [
            job.get_executable().as_posix(),
            *[os.path.expandvars(arg) for arg in job.get_args()],
        ]
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
        )
        self.logger.info(cmd_exec.stderr)
        self.logger.info(cmd_exec.stdout)

        if cmd_exec.returncode != 0:
            self.logger.error(cmd_exec.stderr)
            _send_event(
                ids.EVTYPE_FM_JOB_FAILURE,
                job.get_source(self._ee_id),
                {ids.ERROR_MSG: cmd_exec.stderr},
            )
            raise OSError(
                f"Script {job.get_name()} failed with exception {cmd_exec.stderr}\nOutput: {cmd_exec.stdout}"
            )

    def run_jobs(self, run_path: Path):
        for job in self._step.get_jobs():
            self.logger.info(f"Running command {job.get_name()}")
            _send_event(
                ids.EVTYPE_FM_JOB_START,
                job.get_source(self._ee_id),
            )
            self.run_job(job, run_path)
            _send_event(
                ids.EVTYPE_FM_JOB_SUCCESS,
                job.get_source(self._ee_id),
            )

    def _load_and_dump_input(
        self,
        transmitters: Dict[int, RecordTransmitter],
        runpath: Path,
    ):
        async def transform_input(input_):
            record = await transmitters[input_.get_name()].load()
            await input_.get_transformation().transform_input(
                record=record,
                mime=input_.get_mime(),
                runpath=runpath,
                location=input_.get_path(),
            )

        futures = []
        for input_ in self._step.get_inputs():
            futures.append(transform_input(input_))
        get_event_loop().run_until_complete(asyncio.gather(*futures))

    def run(self, inputs=None):
        async def transform_output(output):
            record = await output.get_transformation().transform_output(
                mime=output.get_mime(),
                location=run_path / output.get_path(),
            )
            transmitter = outputs[output.get_name()]
            await transmitter.transmit_record(record)

        with tempfile.TemporaryDirectory() as run_path:
            run_path = Path(run_path)
            self._load_and_dump_input(transmitters=inputs, runpath=run_path)
            _send_event(
                ids.EVTYPE_FM_STEP_RUNNING,
                self._step.get_source(self._ee_id),
            )

            outputs = {}
            self.run_jobs(run_path)

            futures = []
            for output in self._step.get_outputs():
                if not (run_path / output.get_path()).exists():
                    raise FileNotFoundError(
                        f"Output file {output.get_path()} was not generated!"
                    )

                outputs[output.get_name()] = self._output_transmitters[
                    output.get_name()
                ]
                futures.append(transform_output(output))
            get_event_loop().run_until_complete(asyncio.gather(*futures))
            _send_event(
                ids.EVTYPE_FM_STEP_SUCCESS,
                self._step.get_source(self._ee_id),
            )
        return outputs
