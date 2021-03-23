import asyncio
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
        async def _load(io_name, transmitter):
            record = await transmitter.load()
            return (io_name, record)

        futures = []
        for input_ in self._step.get_inputs():
            futures.append(_load(input_.get_name(), transmitters[input_.get_name()]))
        results = asyncio.get_event_loop().run_until_complete(asyncio.gather(*futures))
        kwargs = {result[0]: result[1].data for result in results}
        function_output = func(**kwargs)

        async def _transmit(io_name, transmitter, data):
            await transmitter.transmit(data)
            return (io_name, transmitter)

        futures = []
        for output in self._step.get_outputs():
            name = output.get_name()
            transmitter = self._output_transmitters[name]
            futures.append(_transmit(name, transmitter, function_output))
        results = asyncio.get_event_loop().run_until_complete(asyncio.gather(*futures))
        transmitter_map = {result[0]: result[1] for result in results}
        return transmitter_map

    def run_job(self, job, transmitters: Dict[str, "RecordTransmitter"], client):
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
