import pathlib
from unittest.mock import MagicMock

import pytest

from ert.config import QueueConfig, QueueSystem
from ert.ensemble_evaluator._builder import (
    EnsembleBuilder,
    LegacyJob,
    LegacyStep,
    RealizationBuilder,
)


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
            .add_step(
                LegacyStep(
                    run_path=pathlib.Path("."),
                    run_arg=MagicMock(),
                    job_script="job_script",
                    job_name="job_name",
                    num_cpu=1,
                    ensemble_config=MagicMock(),
                    jobs=[
                        LegacyJob(
                            ext_job=MagicMock(),
                            id_="4",
                            index="5",
                            name="echo_command",
                        )
                    ],
                    id_="3",
                    name="some_step",
                    max_runtime=0,
                )
            )
            .active(active_real)
        )
        .set_id("1")
    )
    ensemble = ensemble.build()

    real = ensemble.reals[0]
    assert real.active == active_real
