import pathlib
import re
from graphlib import CycleError
from unittest.mock import MagicMock

import pytest

from ert.data import (
    CopyTransformation,
    SerializationTransformation,
    TransformationDirection,
)
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
        .set_legacy_dependencies(MagicMock(), MagicMock())
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
    "steps,expected,ambiguous",
    [
        (
            [
                {
                    "name": "task1",
                    "inputs": ["parameters"],
                    "outputs": ["task1_output"],
                },
                {
                    "name": "task2",
                    "inputs": ["task1_output"],
                    "outputs": ["task2_output"],
                },
                {
                    "name": "task3",
                    "inputs": ["task1_output"],
                    "outputs": ["task3_output"],
                },
                {
                    "name": "task4",
                    "inputs": [
                        "task5_output",
                        "task2_output",
                        "task6_output",
                    ],
                    "outputs": ["task4_output"],
                },
                {
                    "name": "task5",
                    "inputs": ["task3_output"],
                    "outputs": ["task5_output"],
                },
                {"name": "task6", "inputs": [], "outputs": ["task6_output"]},
            ],
            ["task1", "task6", "task2", "task3", "task5", "task4"],
            [],
        ),
        (
            [
                {"name": "a", "inputs": [], "outputs": []},
                {"name": "b", "inputs": [], "outputs": []},
            ],
            [],
            ["a", "b"],
        ),
        (
            [
                {"name": "a", "inputs": [], "outputs": ["a_out"]},
                {"name": "b", "inputs": ["a_out"], "outputs": []},
                {"name": "c", "inputs": [], "outputs": []},
            ],
            ["a", "b"],
            ["c"],
        ),
        (
            [
                {"name": "a", "inputs": [], "outputs": ["a_out", "a_out2"]},
                {"name": "b", "inputs": ["a_out2"], "outputs": ["b_out", "b_out2"]},
                {"name": "c", "inputs": ["a_out", "b_out"], "outputs": []},
            ],
            ["a", "b", "c"],
            [],
        ),
        pytest.param(
            [
                {"name": "step0", "inputs": [], "outputs": ["step0_out"]},
                {"name": "step1", "inputs": ["step0_out"], "outputs": []},
                {"name": "step1", "inputs": ["step0_out"], "outputs": []},
            ],
            [],
            [],
            marks=pytest.mark.xfail(
                raises=ValueError, strict=True, reason="Duplicate step name"
            ),
        ),
        pytest.param(
            [
                {"name": "step0", "inputs": ["step1_out"], "outputs": ["step0_out"]},
                {"name": "step1", "inputs": ["step0_out"], "outputs": ["step1_out"]},
            ],
            [],
            [],
            marks=pytest.mark.xfail(
                raises=CycleError, strict=True, reason="Cyclical relation"
            ),
        ),
    ],
)
def test_topological_sort(steps, expected, ambiguous):
    """Checks that steps are topologically sortable.

    For all ambiguous steps, assert that they are at least present in the
    sorted step. An ambiguous step is an isolated vertex in the topology graph,
    i.e. it does not depend on the input of any other step, nor does any other
    step depend on its output. It is ambiguous because it does not matter where
    in the topological sorting it appears.

    For expected steps, assert that they are equal to the sorted steps, minus
    any ambiguous steps.
    """
    real = RealizationBuilder().set_iens(0).active(True)
    transmitted_factory = MagicMock()
    non_transmitted_factory = MagicMock().return_value = MagicMock()
    non_transmitted_factory.return_value.is_transmitted.return_value = False
    for step_def in steps:
        step = (
            StepBuilder()
            .set_run_path(pathlib.Path("."))
            .set_run_arg(MagicMock())
            .set_job_script("job_script")
            .set_job_name("job_name")
            .set_num_cpu(1)
            .set_callback_arguments(MagicMock())
            .set_done_callback(MagicMock())
            .set_exit_callback(MagicMock())
            .set_id("0")
            .set_name(step_def["name"])
        )
        for input_ in step_def["inputs"]:
            step.add_input(
                InputBuilder()
                .set_name(input_)
                .set_transmitter_factory(transmitted_factory)
                .set_transformation(
                    SerializationTransformation(location=pathlib.Path())
                )
            )
        for output in step_def["outputs"]:
            step.add_output(
                OutputBuilder()
                .set_name(output)
                .set_transmitter_factory(non_transmitted_factory)
                .set_transformation(
                    SerializationTransformation(location=pathlib.Path())
                )
            )
        real.add_step(step)

    ensemble = (
        EnsembleBuilder()
        .set_legacy_dependencies(MagicMock(), MagicMock())
        .add_realization(real)
        .set_id("0")
        .build()
    )
    real = ensemble.reals[0]

    if ambiguous:
        sorted_ = [step.name for step in list(real.get_steps_sorted_topologically())]
        for step in ambiguous:
            assert step in sorted_

    if expected:
        assert expected == [
            step.name
            for step in real.get_steps_sorted_topologically()
            if step.name not in ambiguous
        ]


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
