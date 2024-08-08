import asyncio

import websockets
from cloudevents.conversion import to_json
from cloudevents.http import CloudEvent

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
    client, event_type, source, event_id, data, **extra_attrs
):
    event = CloudEvent(
        {"type": event_type, "source": source, "id": event_id, **extra_attrs}, data
    )
    await client._send(to_json(event))


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

    async def evaluate(self, config, _, __):
        event_id = 0
        await wait_for_evaluator(
            base_url=config.url,
            token=config.token,
            cert=config.cert,
        )
        async with Client(config.url + "/dispatch") as dispatch:
            await send_dispatch_event(
                dispatch,
                identifiers.EVTYPE_ENSEMBLE_STARTED,
                f"/ert/ensemble/{self.id_}",
                f"event-{event_id}",
                None,
            )

            event_id += 1
            for real in range(0, self.test_reals):
                job_failed = False
                await send_dispatch_event(
                    dispatch,
                    identifiers.EVTYPE_REALIZATION_UNKNOWN,
                    f"/ert/ensemble/{self.id_}/real/{real}",
                    f"event-{event_id}",
                    None,
                )
                event_id += 1
                for job in range(0, self.jobs):
                    await send_dispatch_event(
                        dispatch,
                        identifiers.EVTYPE_FORWARD_MODEL_RUNNING,
                        f"/ert/ensemble/{self.id_}/real/{real}/forward_model/{job}",
                        f"event-{event_id}",
                        {"current_memory_usage": 1000},
                    )
                    event_id += 1
                    await send_dispatch_event(
                        dispatch,
                        identifiers.EVTYPE_FORWARD_MODEL_SUCCESS,
                        f"/ert/ensemble/{self.id_}/real/{real}/forward_model/{job}",
                        f"event-{event_id}",
                        {"current_memory_usage": 1000},
                    )
                    event_id += 1
                if job_failed:
                    await send_dispatch_event(
                        dispatch,
                        identifiers.EVTYPE_REALIZATION_FAILURE,
                        f"/ert/ensemble/{self.id_}/real/{real}/forward_model/{job}",
                        f"event-{event_id}",
                        {},
                    )
                    event_id += 1
                else:
                    await send_dispatch_event(
                        dispatch,
                        identifiers.EVTYPE_REALIZATION_SUCCESS,
                        f"/ert/ensemble/{self.id_}/real/{real}/forward_model/{job}",
                        f"event-{event_id}",
                        {},
                    )
                    event_id += 1

            data = self.result if self.result else None
            extra_attrs = {}
            if self.result_datacontenttype:
                extra_attrs["datacontenttype"] = self.result_datacontenttype
            await send_dispatch_event(
                dispatch,
                identifiers.EVTYPE_ENSEMBLE_SUCCEEDED,
                f"/ert/ensemble/{self.id_}",
                f"event-{event_id}",
                data,
                **extra_attrs,
            )

    @property
    def cancellable(self) -> bool:
        return False
