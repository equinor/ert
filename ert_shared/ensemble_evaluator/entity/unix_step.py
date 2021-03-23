import os
import stat
import subprocess
import tempfile

import prefect
from ert_shared.ensemble_evaluator.client import Client
from ert_shared.ensemble_evaluator.entity import identifiers as ids


class UnixTask(prefect.Task):
    def __init__(self, step, output_transmitters, ee_id, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._step = step
        self._output_transmitters = output_transmitters
        self._ee_id = ee_id

    def get_step(self):
        return self._step

    def run_job(self, client, job, run_path):
        shell_cmd = ["python3", job.get_executable(), *job.get_args()]
        cmd_exec = subprocess.run(
            shell_cmd,
            universal_newlines=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=run_path,
        )
        self.logger.info(cmd_exec.stdout)

        if cmd_exec.returncode != 0:
            self.logger.error(cmd_exec.stderr)
            client.send_event(
                ev_type=ids.EVTYPE_FM_JOB_FAILURE,
                ev_source=job.get_source(self._ee_id),
                ev_data={ids.ERROR_MSG: cmd_exec.stderr},
            )
            raise OSError(
                f"Script {job.get_name()} failed with exception {cmd_exec.stderr}"
            )

    def run_jobs(self, client, run_path):
        for job in self._step.get_jobs():
            self.logger.info(f"Running command {job.get_name()}")
            client.send_event(
                ev_type=ids.EVTYPE_FM_JOB_START,
                ev_source=job.get_source(self._ee_id),
            )
            self.run_job(client, job, run_path)
            client.send_event(
                ev_type=ids.EVTYPE_FM_JOB_SUCCESS,
                ev_source=job.get_source(self._ee_id),
            )

    def _load_and_dump_input(self, transmitters, runpath):
        for input_ in self._step.get_inputs():
            # TODO: use Path
            transmitters[input_.get_name()].dump(
                os.path.join(runpath, input_.get_path()), input_.get_mime()
            )
            if input_.is_executable():
                path = os.path.join(runpath, input_.get_path())
                st = os.stat(path)
                os.chmod(path, st.st_mode | stat.S_IEXEC)

    def run(self, inputs=None):
        with tempfile.TemporaryDirectory() as run_path:
            self._load_and_dump_input(transmitters=inputs, runpath=run_path)
            with Client(self._step.get_ee_url()) as ee_client:
                ee_client.send_event(
                    ev_type=ids.EVTYPE_FM_STEP_START,
                    ev_source=self._step.get_source(self._ee_id),
                )

                outputs = {}
                self.run_jobs(ee_client, run_path)

                for output in self._step.get_outputs():
                    if not os.path.exists(os.path.join(run_path, output.get_path())):
                        raise FileNotFoundError(
                            f"Output file {output.get_path()} was not generated!"
                        )

                    outputs[output.get_name()] = self._output_transmitters[
                        output.get_name()
                    ]
                    outputs[output.get_name()].transmit(
                        os.path.join(run_path, output.get_path())
                    )

                ee_client.send_event(
                    ev_type=ids.EVTYPE_FM_STEP_SUCCESS,
                    ev_source=self._step.get_source(self._ee_id),
                )
        return outputs
