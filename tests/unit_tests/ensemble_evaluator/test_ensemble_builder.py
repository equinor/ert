import pathlib
import re
from unittest.mock import MagicMock

import pytest

from ert.config import QueueConfig, QueueSystem
from ert.data import CopyTransformation, TransformationDirection
from ert.ensemble_evaluator._builder import (
    EnsembleBuilder,
    InputBuilder,
    OutputBuilder,
    RealizationBuilder,
    StepBuilder,
)
from ert.ensemble_evaluator._builder._job import LegacyJobBuilder


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
                StepBuilder()
                .set_run_path(pathlib.Path("."))
                .set_run_arg(MagicMock())
                .set_job_script("job_script")
                .set_job_name("job_name")
                .set_num_cpu(1)
                .set_callback_arguments(MagicMock())
                .set_done_callback(MagicMock())
                .set_exit_callback(MagicMock())
                .add_job(
                    LegacyJobBuilder()
                    .set_ext_job(MagicMock())
                    .set_id("4")
                    .set_index("5")
                    .set_name("echo_command")
                )
                .set_id("3")
                .set_name("some_step")
                .set_dummy_io()
            )
            .active(active_real)
        )
        .set_id("1")
    )
    ensemble = ensemble.build()

    real = ensemble.reals[0]
    assert real.active == active_real
    assert real.source() == "/ert/ensemble/1/real/2"


@pytest.mark.parametrize(
    "builder,transformation",
    [
        pytest.param(
            InputBuilder(),
            CopyTransformation(pathlib.Path("foo"), TransformationDirection.TO_RECORD),
            marks=pytest.mark.raises(
                exception=ValueError,
                match=(
                    ".+does not allow 'from_record', only "
                    + f"'{TransformationDirection.TO_RECORD}'"
                ),
                match_flags=(re.MULTILINE | re.DOTALL),
            ),
        ),
        pytest.param(
            OutputBuilder(),
            CopyTransformation(
                pathlib.Path("foo"), TransformationDirection.FROM_RECORD
            ),
            marks=pytest.mark.raises(
                exception=ValueError,
                match=(
                    ".+does not allow 'to_record', only "
                    + f"'{TransformationDirection.FROM_RECORD}'"
                ),
                match_flags=(re.MULTILINE | re.DOTALL),
            ),
        ),
        pytest.param(
            InputBuilder(),
            CopyTransformation(pathlib.Path("foo")),
        ),
        pytest.param(
            OutputBuilder(),
            CopyTransformation(pathlib.Path("foo")),
        ),
    ],
)
def test_io_direction_constraints(builder, transformation):
    builder.set_transformation(transformation)
