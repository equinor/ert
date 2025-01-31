from ert.config import QueueConfig
from ert.ensemble_evaluator import Ensemble
from ert.ensemble_evaluator._ensemble import ForwardModelStep, Realization


class TestEnsemble(Ensemble):
    __test__ = False

    def __init__(self, iter_, reals, fm_steps, id_):
        the_reals = [
            Realization(
                real_no,
                fm_steps=[
                    ForwardModelStep(str(fm_idx), "") for fm_idx in range(fm_steps)
                ],
                active=True,
                max_runtime=0,
                num_cpu=0,
                run_arg=None,
                job_script=None,
                realization_memory=0,
            )
            for real_no in range(reals)
        ]
        super().__init__(the_reals, {}, QueueConfig(), 0, id_)

    async def evaluate(self, config, _, __):
        pass

    @property
    def cancellable(self) -> bool:
        return False
