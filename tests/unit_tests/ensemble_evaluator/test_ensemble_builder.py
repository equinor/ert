from unittest.mock import MagicMock

import pytest

from ert.config import ForwardModelStep, QueueConfig, QueueSystem
from ert.ensemble_evaluator._builder import Realization
from ert.ensemble_evaluator._builder._legacy import LegacyEnsemble


@pytest.mark.parametrize("active_real", [True, False])
@pytest.mark.usefixtures("using_scheduler")
def test_build_ensemble(active_real):
    ensemble = LegacyEnsemble(
        [
            Realization(
                iens=2,
                run_arg=MagicMock(),
                num_cpu=1,
                max_runtime=0,
                job_script="job_script",
                forward_models=[ForwardModelStep("echo_command", "")],
                active=active_real,
            )
        ],
        {},
        QueueConfig(queue_system=QueueSystem.LOCAL),
        0,
        "1",
    )

    real = ensemble.reals[0]
    assert real.active == active_real
