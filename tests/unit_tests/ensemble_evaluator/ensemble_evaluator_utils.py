import asyncio
from datetime import datetime

import orjson
import websockets

from _ert.async_utils import new_event_loop
from _ert_forward_model_runner.client import Client
from ert.config import QueueConfig
from ert.ensemble_evaluator import Ensemble, identifiers
from ert.ensemble_evaluator._ensemble import ForwardModelStep, Realization
from ert.ensemble_evaluator._wait_for_evaluator import wait_for_evaluator


def _mock_ws(host, port, messages, delay_startup=0):
    loop = new_event_loop()
    done = loop.create_future()

    async def _handler(websocket, path):
        while True:
            msg = await websocket.recv()
            messages.append(msg)
            if msg == "stop":
                done.set_result(None)
                break

    async def _run_server():
        await asyncio.sleep(delay_startup)
        async with websockets.server.serve(
            _handler, host, port, ping_timeout=1, ping_interval=1
        ):
            await done

    loop.run_until_complete(_run_server())
    loop.close()


async def send_dispatch_event(
    client, type, id, ensemble, data, real=None, fm_step=None
):
    event = {
        "type": type,
        "time": datetime.now(),
        "id": id,
        "ensemble": str(ensemble),
        "data": data,
    }
    if real is not None:
        event["real"] = str(real)
    if fm_step is not None:
        event["fm_step"] = str(fm_step)

    await client._send(orjson.dumps(event))


class TestEnsemble(Ensemble):
    __test__ = False

    def __init__(self, _iter, reals, jobs, id_):
        self.iter = _iter
        self.test_reals = reals
        self.jobs = jobs
        self.result = None

        the_reals = [
            Realization(
                real_no,
                forward_models=[
                    ForwardModelStep(str(fm_idx), "") for fm_idx in range(0, jobs)
                ],
                active=True,
                max_runtime=0,
                num_cpu=0,
                run_arg=None,
                job_script=None,
                realization_memory=0,
            )
            for real_no in range(0, reals)
        ]
        super().__init__(the_reals, {}, QueueConfig(), 0, id_)

    async def evaluate(self, config):
        event_id = 0
        await wait_for_evaluator(
            base_url=config.url,
            token=config.token,
            cert=config.cert,
        )
        async with Client(config.url + "/dispatch") as dispatch:
            await send_dispatch_event(
                client=dispatch,
                type=identifiers.EVTYPE_ENSEMBLE_STARTED,
                id="event1",
                ensemble=self.id_,
                data=None,
            )

            event_id += 1
            for real in range(0, self.test_reals):
                job_failed = False

                await send_dispatch_event(
                    client=dispatch,
                    type=identifiers.EVTYPE_REALIZATION_UNKNOWN,
                    id=f"event-{event_id}",
                    real=real,
                    ensemble=self.id_,
                    data=None,
                )
                event_id += 1
                for job in range(0, self.jobs):
                    await send_dispatch_event(
                        client=dispatch,
                        type=identifiers.EVTYPE_FORWARD_MODEL_RUNNING,
                        id=f"event-{event_id}",
                        real=real,
                        ensemble=self.id_,
                        fm_step=job,
                        data={"current_memory_usage": 1000},
                    )
                    event_id += 1
                    await send_dispatch_event(
                        client=dispatch,
                        type=identifiers.EVTYPE_FORWARD_MODEL_SUCCESS,
                        id=f"event-{event_id}",
                        real=real,
                        ensemble=self.id_,
                        fm_step=job,
                        data={"current_memory_usage": 1000},
                    )
                    event_id += 1
                if job_failed:
                    await send_dispatch_event(
                        client=dispatch,
                        type=identifiers.EVTYPE_REALIZATION_FAILURE,
                        id=f"event-{event_id}",
                        real=real,
                        ensemble=self.id_,
                        fm_step=job,
                        data={},
                    )
                    event_id += 1
                else:
                    await send_dispatch_event(
                        client=dispatch,
                        type=identifiers.EVTYPE_REALIZATION_SUCCESS,
                        id=f"event-{event_id}",
                        real=real,
                        ensemble=self.id_,
                        fm_step=job,
                        data={},
                    )
                    event_id += 1

            data = self.result if self.result else None
            await send_dispatch_event(
                client=dispatch,
                type=identifiers.EVTYPE_ENSEMBLE_SUCCEEDED,
                id=f"event-{event_id}",
                ensemble=self.id_,
                data=data,
            )

    @property
    def cancellable(self) -> bool:
        return False
