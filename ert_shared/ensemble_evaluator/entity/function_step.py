from typing import Dict, Optional
from prefect import Task
import prefect
from ert_shared.ensemble_evaluator.client import Client
from ert_shared.ensemble_evaluator.entity import identifiers as ids


class FunctionTask(prefect.Task):
    def __init__(self, step, output_transmitters, ee_id, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._step = step
        self._output_transmitters = output_transmitters
        self._ee_id = ee_id

    def _attempt_execute(self, *, func, transmitters):
        kwargs = {
            input_.get_name(): transmitters[input_.get_name()].load().data
            for input_ in self._step.get_inputs()
        }
        function_output = func(**kwargs)

        transmitter_map = {}
        for output in self._step.get_outputs():
            name = output.get_name()
            transmitter = self._output_transmitters[name]
            transmitter.transmit(function_output)
            transmitter_map[name] = transmitter
        return transmitter_map

    def run_job(
        self, job, transmitters: Dict[str, "RecordTransmitter"], client
    ):
        self.logger.info(f"Running function {job.get_name()}")
        client.send_event(
            ev_type=ids.EVTYPE_FM_JOB_START,
            ev_source=job.get_source(self._ee_id),
        )
        try:
            output = self._attempt_execute(
                func=job.get_command(), transmitters=transmitters
            )
        except Exception as e:
            self.logger.error(str(e))
            client.send_event(
                ev_type=ids.EVTYPE_FM_JOB_FAILURE,
                ev_source=job.get_source(self._ee_id),
                ev_data={ids.ERROR_MSG: str(e)},
            )
            raise e
        else:
            client.send_event(
                ev_type=ids.EVTYPE_FM_JOB_SUCCESS,
                ev_source=job.get_source(self._ee_id),
            )
        return output

    def run(self, inputs: Dict[str, "ert3.data.RecordTransmitter"]):
        with Client(self._step.get_ee_url()) as ee_client:
            ee_client.send_event(
                ev_type=ids.EVTYPE_FM_STEP_START,
                ev_source=self._step.get_source(self._ee_id),
            )

            output = self.run_job(
                job=self._step.get_jobs()[0], transmitters=inputs, client=ee_client
            )

            ee_client.send_event(
                ev_type=ids.EVTYPE_FM_STEP_SUCCESS,
                ev_source=self._step.get_source(self._ee_id),
            )

        return output
