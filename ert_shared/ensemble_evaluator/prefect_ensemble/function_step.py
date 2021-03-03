from prefect import Task
from ert_shared.ensemble_evaluator.prefect_ensemble.client import Client
from ert_shared.ensemble_evaluator.entity import identifiers as ids
from ert_shared.ensemble_evaluator.prefect_ensemble.storage_driver import (
    storage_driver_factory,
)


class FunctionStep(Task):
    def __init__(
        self,
        step,
        url,
        ee_id,
        storage_config,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._step = step
        self._url = url
        self._ee_id = ee_id
        self._storage = storage_driver_factory(storage_config, ".")

    def get_iens(self):
        return self._step[ids.IENS]

    def get_stage_id(self):
        return self._step[ids.STAGE_ID]

    def get_step_id(self):
        return self._step[ids.STEP_ID]

    def get_ee_id(self):
        return self._ee_id

    @property
    def job(self):
        return self._step[ids.JOBS][0]

    @property
    def step_source(self):
        iens = self.get_iens()
        stage_id = self.get_stage_id()
        step_id = self.get_step_id()
        return f"/ert/ee/{self._ee_id}/real/{iens}/stage/{stage_id}/step/{step_id}"

    def function_source(self, fun_id):
        return f"{self.step_source}/job/{fun_id}"

    def run_job(self, client, job):
        try:
            result = job["executable"](**self._step["step_input"])
            # Store the results
            return self._storage.store_data(result, job["output"], self.get_iens())
        except Exception as e:
            self.logger.error(str(e))
            client.send_event(
                ev_type=ids.EVTYPE_FM_JOB_FAILURE,
                ev_source=self.function_source(job[ids.ID]),
                ev_data={"stderr": str(e)},
            )
            raise e

    def run_jobs(self, client):
        self.logger.info(f"Running function {self.job[ids.NAME]}")
        client.send_event(
            ev_type=ids.EVTYPE_FM_JOB_START,
            ev_source=self.function_source(self.job[ids.ID]),
        )
        output = self.run_job(client, self.job)
        client.send_event(
            ev_type=ids.EVTYPE_FM_JOB_SUCCESS,
            ev_source=self.function_source(self.job[ids.ID]),
        )
        return output

    def run(self, expected_res=None):
        with Client(self._url) as ee_client:
            ee_client.send_event(
                ev_type=ids.EVTYPE_FM_STEP_START,
                ev_source=self.step_source,
            )

            output = self.run_jobs(ee_client)

            ee_client.send_event(
                ev_type=ids.EVTYPE_FM_STEP_SUCCESS,
                ev_source=self.step_source,
            )

        return {ids.IENS: self.get_iens(), ids.OUTPUTS: [output]}
