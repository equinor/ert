import os
import subprocess
from prefect import Task
from ert_shared.ensemble_evaluator.prefect_ensemble.client import Client
from ert_shared.ensemble_evaluator.prefect_ensemble.storage_driver import (
    storage_driver_factory,
)
from cloudevents.http import to_json, CloudEvent
from ert_shared.ensemble_evaluator.entity import identifiers as ids


class FunctionStep(Task):
    def __init__(
        self,
        resources,
        source,
        outputs,
        iens,
        url,
        function_id,
        step_id,
        stage_id,
        ee_id,
        run_path,
        storage_config,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._resources = resources
        self._source = source
        self._outputs = outputs
        self._iens = iens
        self._url = url
        self._function_id = function_id
        self._step_id = step_id
        self._stage_id = stage_id
        self._ee_id = ee_id
        self._run_path = run_path
        self._storage_config = storage_config

    def get_iens(self):
        return self._iens

    def get_function_id(self):
        return self._function_id

    def get_stage_id(self):
        return self._stage_id

    def get_step_id(self):
        return self._step_id

    def get_ee_id(self):
        return self._ee_id

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
        os.makedirs(run_path, exist_ok=True)
        return storage_driver_factory(self._storage_config, run_path)

    def evaluate_function(self, client, run_path):
        self.logger.info(f"Evaluating function")
        event = CloudEvent(
            {
                "type": ids.EVTYPE_FM_JOB_START,
                "source": f"/ert/ee/{self._ee_id}/real/{self._iens}/stage/{self._stage_id}/step/{self._step_id}/job/{self._function_id}",
                "datacontenttype": "application/json",
            },
        )
        client.send(to_json(event).decode())
        try:
            inputs = [1, 4, 5, 6, 7, 2, 1, 2, 4, 6, 7, 8, 9]
            print(inputs)
            with open(self._source, "r") as f:
                exec(compile(f.read(), "function", "exec"))
            print(inputs)
        except Exception as e:
            self.logger.error(str(e))
            event = CloudEvent(
                {
                    "type": ids.EVTYPE_FM_JOB_FAILURE,
                    "source": f"/ert/ee/{self._ee_id}/real/{self._iens}/stage/{self._stage_id}/step/{self._step_id}/job/{self._function_id}",
                    "datacontenttype": "application/json",
                },
                {"stderr": str(e)},
            )
            client.send(to_json(event).decode())
            raise e

        event = CloudEvent(
            {
                "type": ids.EVTYPE_FM_JOB_SUCCESS,
                "source": f"/ert/ee/{self._ee_id}/real/{self._iens}/stage/{self._stage_id}/step/{self._step_id}/job/{self._function_id}",
                "datacontenttype": "application/json",
            },
        )

        client.send(to_json(event).decode())

    def run(self, expected_res=None):
        run_path = os.path.join(self._run_path, str(self._iens))
        storage = self.storage_driver(run_path)
        self.retrieve_resources(expected_res, storage)

        with Client(self._url) as ee_client:
            event = CloudEvent(
                {
                    "type": ids.EVTYPE_FM_STEP_START,
                    "source": f"/ert/ee/{self._ee_id}/real/{self._iens}/stage/{self._stage_id}/step/{self._step_id}",
                    "datacontenttype": "application/json",
                },
            )
            ee_client.send(to_json(event).decode())

            outputs = []
            self.evaluate_function(ee_client, run_path)

            for output in self._outputs:
                if not os.path.exists(os.path.join(run_path, output)):
                    raise FileNotFoundError(f"Output file {output} was not generated!")

                outputs.append(storage.store(output, self._iens))

            event = CloudEvent(
                {
                    "type": ids.EVTYPE_FM_STEP_SUCCESS,
                    "source": f"/ert/ee/{self._ee_id}/real/{self._iens}/stage/{self._stage_id}/step/{self._step_id}",
                    "datacontenttype": "application/json",
                },
            )
            ee_client.send(to_json(event).decode())
        return {"iens": self._iens, "outputs": outputs}
