import subprocess
from pathlib import Path
from prefect import Task
from ert_shared.ensemble_evaluator.prefect_ensemble.client import Client
from ert_shared.ensemble_evaluator.prefect_ensemble.storage_driver import (
    storage_driver_factory,
)
from ert_shared.ensemble_evaluator.entity import identifiers as ids


class UnixStep(Task):
    def __init__(
        self,
        step,
        resources,
        cmd,
        url,
        ee_id,
        run_path,
        storage_config,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._step = step
        self._resources = resources
        self._cmd = cmd
        self._url = url
        self._ee_id = ee_id
        self._run_path = Path(run_path)
        self._storage_config = storage_config

    def get_iens(self):
        return self._step[ids.IENS]

    def get_stage_id(self):
        return self._step[ids.STAGE_ID]

    def get_step_id(self):
        return self._step[ids.STEP_ID]

    def get_ee_id(self):
        return self._ee_id

    @property
    def jobs(self):
        return self._step[ids.JOBS]

    @property
    def step_source(self):
        iens = self.get_iens()
        stage_id = self.get_stage_id()
        step_id = self.get_step_id()
        return f"/ert/ee/{self._ee_id}/real/{iens}/stage/{stage_id}/step/{step_id}"

    def job_source(self, job_id):
        return f"{self.step_source}/job/{job_id}"

    def retrieve_resources(self, expected_res, storage):
        if not expected_res:
            resources = self._resources
        else:
            resources = [
                item for sublist in expected_res for item in sublist
            ] + self._resources
        for resource in resources:
            storage.retrieve(resource)

    def storage_driver(self, run_path):
        Path(run_path).mkdir(parents=True, exist_ok=True)
        return storage_driver_factory(self._storage_config, run_path)

    def run_job(self, client, job, run_path):
        shell_cmd = [self._cmd, job[ids.EXECUTABLE], *job[ids.ARGS]]
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
                ev_source=self.job_source(job[ids.ID]),
                ev_data={ids.ERROR_MSG: cmd_exec.stderr},
            )
            raise OSError(
                f"Script {job[ids.NAME]} failed with exception {cmd_exec.stderr}"
            )

    def run_jobs(self, client, run_path):
        for job in self.jobs:
            self.logger.info(f"Running command {self._cmd}  {job[ids.NAME]}")
            client.send_event(
                ev_type=ids.EVTYPE_FM_JOB_START,
                ev_source=self.job_source(job[ids.ID]),
            )
            self.run_job(client, job, run_path)
            client.send_event(
                ev_type=ids.EVTYPE_FM_JOB_SUCCESS,
                ev_source=self.job_source(job[ids.ID]),
            )

    def run(self, expected_res=None):
        run_path = self._run_path / str(self.get_iens())
        storage = self.storage_driver(run_path)
        self.retrieve_resources(expected_res, storage)

        with Client(self._url) as ee_client:
            ee_client.send_event(
                ev_type=ids.EVTYPE_FM_STEP_START,
                ev_source=self.step_source,
            )

            outputs = []
            self.run_jobs(ee_client, run_path)

            for output in self._step[ids.OUTPUTS]:
                if not (run_path / output).exists():
                    raise FileNotFoundError(f"Output file {output} was not generated!")

                outputs.append(storage.store(output, self.get_iens()))

            ee_client.send_event(
                ev_type=ids.EVTYPE_FM_STEP_SUCCESS,
                ev_source=self.step_source,
            )
        return {ids.IENS: self.get_iens(), ids.OUTPUTS: outputs}
