import asyncio

import websockets

from _ert.async_utils import new_event_loop
from ert.config import QueueConfig
from ert.ensemble_evaluator import Ensemble
from ert.ensemble_evaluator._ensemble import ForwardModelStep, Realization


def _mock_ws(host, port, messages, delay_startup=0):
    loop = new_event_loop()
    done = loop.create_future()

    async def _handler(websocket):
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


class TestEnsemble(Ensemble):
    __test__ = False

    def __init__(self, iter_, reals, fm_steps, id_):
        self.iter = iter_
        self.test_reals = reals
        self.fm_steps = fm_steps

        the_reals = [
            Realization(
                real_no,
                fm_steps=[
                    ForwardModelStep(str(fm_idx), "") for fm_idx in range(0, fm_steps)
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
        pass

    @property
    def cancellable(self) -> bool:
        return False
