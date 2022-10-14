import asyncio
import pickle
from typing import TYPE_CHECKING, Any, Callable, Dict, Tuple

import prefect

from _ert_job_runner.client import Client
from ert.async_utils import get_event_loop
from ert.data import BlobRecord, NumericalRecord, Record, record_data
from ert.ensemble_evaluator import identifiers as ids

from ._io_ import IO
from ._io_map import _stage_transmitter_mapping
from ._job import FunctionJob

if TYPE_CHECKING:
    import ert

    from ._step import Step


class FunctionTask(prefect.Task):  # type: ignore
    def __init__(
        self,
        step: "Step",
        output_transmitters: _stage_transmitter_mapping,
        ens_id: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.step = step
        self._output_transmitters = output_transmitters
        self._ens_id = ens_id

    def _attempt_execute(
        self,
        *,
        func: Callable[..., Any],
        transmitters: _stage_transmitter_mapping,
    ) -> _stage_transmitter_mapping:
        async def _load(
            io_: IO, transmitter: "ert.data.RecordTransmitter"
        ) -> Tuple[str, Record]:
            record = await transmitter.load()
            return (io_.name, record)

        input_futures = []
        for input_ in self.step.inputs:
            transmitter = transmitters[input_.name]
            if transmitter:
                input_futures.append(_load(input_, transmitter))
            else:
                self.logger.info("no transmitter for input %s", input_.name)
        results = get_event_loop().run_until_complete(asyncio.gather(*input_futures))
        kwargs = {result[0]: result[1].data for result in results}
        function_output: Dict[str, record_data] = func(**kwargs)

        async def _transmit(
            io_: IO, transmitter: "ert.data.RecordTransmitter", data: record_data
        ) -> Tuple[str, "ert.data.RecordTransmitter"]:
            record: Record = (
                BlobRecord(data=data)
                if isinstance(data, bytes)
                else NumericalRecord(data=data)
            )
            await transmitter.transmit_record(record)
            return (io_.name, transmitter)

        output_futures = []
        for output in self.step.outputs:
            transmitter = self._output_transmitters[output.name]
            if transmitter:
                output_futures.append(
                    _transmit(output, transmitter, function_output[output.name])
                )
            else:
                self.logger.info("no transmitter for output %s", output.name)
        results = get_event_loop().run_until_complete(asyncio.gather(*output_futures))
        transmitter_map: _stage_transmitter_mapping = {
            result[0]: result[1] for result in results
        }
        return transmitter_map

    def run_job(
        self,
        job: FunctionJob,
        transmitters: _stage_transmitter_mapping,
        client: Client,
    ) -> _stage_transmitter_mapping:
        self.logger.info(f"Running function {job.name}")
        client.send_event(
            ev_type=ids.EVTYPE_FM_JOB_START,
            ev_source=job.source(),
        )
        try:
            function: Callable[..., Any] = pickle.loads(job.command)
            output = self._attempt_execute(func=function, transmitters=transmitters)
        except Exception as e:
            self.logger.error(str(e))
            client.send_event(
                ev_type=ids.EVTYPE_FM_JOB_FAILURE,
                ev_source=job.source(),
                ev_data={ids.ERROR_MSG: str(e)},
            )
            raise e
        else:
            client.send_event(
                ev_type=ids.EVTYPE_FM_JOB_SUCCESS,
                ev_source=job.source(),
            )
        return output

    def run(self, inputs: _stage_transmitter_mapping):  # type: ignore  # pylint: disable=arguments-differ  # noqa
        with Client(
            prefect.context.url,  # type: ignore  # pylint: disable=no-member
            prefect.context.token,  # type: ignore  # pylint: disable=no-member
            prefect.context.cert,  # type: ignore  # pylint: disable=no-member
        ) as ee_client:
            ee_client.send_event(
                ev_type=ids.EVTYPE_FM_STEP_RUNNING,
                ev_source=self.step.source(),
            )

            job = self.step.jobs[0]
            if not isinstance(job, FunctionJob):
                raise TypeError(f"unexpected job {type(job)} in function task")

            output = self.run_job(job=job, transmitters=inputs, client=ee_client)

            ee_client.send_event(
                ev_type=ids.EVTYPE_FM_STEP_SUCCESS,
                ev_source=self.step.source(),
            )

        return output
