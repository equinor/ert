from unittest.mock import MagicMock

import pytest

from ert.config import ForwardModel, QueueConfig, QueueSystem
from ert.ensemble_evaluator._builder import (
    EnsembleBuilder,
    RealizationBuilder,
)


@pytest.mark.xfail(reason="Needs reimplementation")
@pytest.mark.parametrize("active_real", [True, False])
def test_build_ensemble(active_real):
    ensemble = (
        EnsembleBuilder()
        .set_legacy_dependencies(
            QueueConfig(queue_system=QueueSystem.LOCAL), MagicMock()
        )
        .add_realization(
            RealizationBuilder()
            .set_iens(2)
            .set_run_arg(MagicMock())
            .set_num_cpu(1)
            .set_max_runtime(0)
            .set_job_script("job_script")
            .set_forward_models([ForwardModel("echo_command", "")])
            .active(active_real)
        )
        .set_id("1")
    )
    ensemble = ensemble.build()

    real = ensemble.reals[0]
    assert real.active == active_real
