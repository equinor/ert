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
        return self._step["iens"]

    def get_stage_id(self):
        return self._step["stage_id"]

    def get_step_id(self):
        return self._step["step_id"]

    def get_ee_id(self):
        return self._ee_id

    @property
    def job(self):
        return self._step["jobs"][0]

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
                ev_source=self.function_source(job["id"]),
                ev_data={"stderr": str(e)},
            )
            raise e

    def run_jobs(self, client):
        self.logger.info(f"Running function {self.job['name']}")
        client.send_event(
            ev_type=ids.EVTYPE_FM_JOB_START,
            ev_source=self.function_source(self.job["id"]),
        )
        output = self.run_job(client, self.job)
        client.send_event(
            ev_type=ids.EVTYPE_FM_JOB_SUCCESS,
            ev_source=self.function_source(self.job["id"]),
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

        return {"iens": self.get_iens(), "outputs": [output]}
