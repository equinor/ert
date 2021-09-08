import asyncio
import pickle
from typing import Dict
import prefect
from ert_shared.ensemble_evaluator.client import Client
from ert_shared.ensemble_evaluator.entity import identifiers as ids

from ert.data import RecordTransmitter, NumericalRecord, BlobRecord, Record


class FunctionTask(prefect.Task):
    def __init__(self, step, output_transmitters, ee_id, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._step = step
        self._output_transmitters = output_transmitters
        self._ee_id = ee_id

    def get_step(self):
        return self._step

    def _attempt_execute(self, *, func, transmitters):
        async def _load(io_, transmitter):
            record = await transmitter.load()
            return (io_.get_name(), record)

        futures = []
        for input_ in self._step.get_inputs():
            futures.append(_load(input_, transmitters[input_.get_name()]))
        results = asyncio.get_event_loop().run_until_complete(asyncio.gather(*futures))
        kwargs = {result[0]: result[1].data for result in results}
        function_output = func(**kwargs)

        async def _transmit(io_, transmitter: RecordTransmitter, data):
            record: Record = (
                BlobRecord(data=data)
                if isinstance(data, bytes)
                else NumericalRecord(data=data)
            )
            await transmitter.transmit_record(record)
            return (io_.get_name(), transmitter)

        futures = []
        for output in self._step.get_outputs():
            transmitter = self._output_transmitters[output.get_name()]
            futures.append(_transmit(output, transmitter, function_output))
        results = asyncio.get_event_loop().run_until_complete(asyncio.gather(*futures))
        transmitter_map = {result[0]: result[1] for result in results}
        return transmitter_map

    def run_job(self, job, transmitters: Dict[str, RecordTransmitter], client):
        self.logger.info(f"Running function {job.get_name()}")
        client.send_event(
            ev_type=ids.EVTYPE_FM_JOB_START,
            ev_source=job.get_source(self._ee_id),
        )
        try:
            function = pickle.loads(job.get_command())
            output = self._attempt_execute(func=function, transmitters=transmitters)
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

    def run(self, inputs: Dict[str, RecordTransmitter]):  # type: ignore
        with Client(
            prefect.context.url, prefect.context.token, prefect.context.cert
        ) as ee_client:
            ee_client.send_event(
                ev_type=ids.EVTYPE_FM_STEP_RUNNING,
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
