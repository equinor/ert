import asyncio

import websockets

from _ert.async_utils import new_event_loop
from _ert.events import (
    EnsembleStarted,
    EnsembleSucceeded,
    ForwardModelStepRunning,
    ForwardModelStepSuccess,
    RealizationSuccess,
    RealizationUnknown,
    event_to_json,
)
from _ert.forward_model_runner.client import Client
from ert.config import QueueConfig
from ert.ensemble_evaluator import Ensemble
from ert.ensemble_evaluator._ensemble import ForwardModelStep, Realization


class TestEnsemble(Ensemble):
    __test__ = False

    def __init__(self, _iter, reals, fm_steps, id_):
        self.iter = _iter
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
        event_id = 0
        async with Client(
            config.get_connection_info().router_uri,
            cert=config.cert,
            token=config.token,
            max_retries=1,
            dealer_name="eval_dispatch",
        ) as dispatch:
            event = EnsembleStarted(ensemble=self.id_)
            await dispatch._send(event_to_json(event))

            event_id += 1
            for real in range(0, self.test_reals):
                real = str(real)

                event = RealizationUnknown(ensemble=self.id_, real=real)
                await dispatch._send(event_to_json(event))

                event_id += 1
                for fm_step in range(0, self.fm_steps):
                    fm_step = str(fm_step)

                    event = ForwardModelStepRunning(
                        ensemble=self.id_,
                        real=real,
                        fm_step=fm_step,
                        current_memory_usage=1000,
                    )
                    await dispatch._send(event_to_json(event))
                    event_id += 1

                    event = ForwardModelStepSuccess(
                        ensemble=self.id_,
                        real=real,
                        fm_step=fm_step,
                        current_memory_usage=1000,
                    )
                    await dispatch._send(event_to_json(event))
                    event_id += 1
                    event_id += 1

                event = RealizationSuccess(ensemble=self.id_, real=real)
                await dispatch._send(event_to_json(event))
                event_id += 1

            event = EnsembleSucceeded(ensemble=self.id_)
            await dispatch._send(event_to_json(event))

    @property
    def cancellable(self) -> bool:
        return False
