import pathlib
import re
from graphlib import CycleError
from unittest.mock import MagicMock, Mock

import ert_shared.ensemble_evaluator.ensemble.builder as ee
import pytest
from ert.data import (
    SerializationTransformation,
    CopyTransformation,
    EclSumTransformation,
    TransformationDirection,
)


@pytest.mark.parametrize("active_real", [True, False])
def test_build_ensemble(active_real):
    ensemble = ee.create_ensemble_builder().add_realization(
        ee.create_realization_builder()
        .set_iens(0)
        .add_step(
            ee.create_step_builder()
            .add_job(
                ee.create_legacy_job_builder()
                .set_id(0)
                .set_name("echo_command")
                .set_ext_job(Mock())
            )
            .set_id("0")
            .set_name("some_step")
            .set_dummy_io()
            .set_type("unix")
        )
        .active(active_real)
    )
    ensemble = ensemble.build()
    real = ensemble.get_reals()[0]
    assert real.is_active() == active_real


def test_build_ensemble_legacy():

    run_context = MagicMock()
    run_context.__len__.return_value = 1
    run_context.is_active = lambda i: True if i == 0 else False

    ext_job = MagicMock()
    ext_job.get_executable = MagicMock(return_value="junk.exe")
    ext_job.name = MagicMock(return_value="junk")
    ext_job.get_arglist = MagicMock(return_value=("arg1", "arg2", "arg3"))

    forward_model = MagicMock()
    forward_model.__len__.return_value = 1
    forward_model.iget_job = lambda i: ext_job if i == 0 else None

    analysis_config = MagicMock()
    analysis_config.get_max_runtime = MagicMock(return_value=0)

    queue_config = MagicMock()

    res_config = MagicMock()

    ensemble_builder = ee.create_ensemble_builder_from_legacy(
        run_context=run_context,
        forward_model=forward_model,
        queue_config=queue_config,
        analysis_config=analysis_config,
        res_config=res_config,
    )

    ensemble = ensemble_builder.build()

    real = ensemble.get_reals()[0]
    assert real.is_active()


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
    real = ee.create_realization_builder().set_iens(0).active(True)
    transmitted_factory = MagicMock()
    non_transmitted_factory = MagicMock().return_value = MagicMock()
    non_transmitted_factory.return_value.is_transmitted.return_value = False
    for step_def in steps:
        step = (
            ee.create_step_builder()
            .set_id("0")
            .set_name(step_def["name"])
            .set_type("unix")
        )
        for input_ in step_def["inputs"]:
            step.add_input(
                ee.create_input_builder()
                .set_name(input_)
                .set_transmitter_factory(transmitted_factory)
                .set_transformation(
                    SerializationTransformation(location=pathlib.Path())
                )
            )
        for output in step_def["outputs"]:
            step.add_output(
                ee.create_output_builder()
                .set_name(output)
                .set_transmitter_factory(non_transmitted_factory)
                .set_transformation(
                    SerializationTransformation(location=pathlib.Path())
                )
            )
        real.add_step(step)

    ensemble = ee.create_ensemble_builder().add_realization(real).build()
    real = ensemble.get_reals()[0]

    if ambiguous:
        sorted_ = [
            step.get_name() for step in list(real.get_steps_sorted_topologically())
        ]
        for step in ambiguous:
            assert step in sorted_

    if expected:
        assert expected == [
            step.get_name()
            for step in real.get_steps_sorted_topologically()
            if step.get_name() not in ambiguous
        ]


def test_io_transformation_required_for_unix():
    with pytest.raises(ValueError, match="has no transformation"):
        (
            ee.create_step_builder()
            .add_input(ee.create_input_builder().set_name("input"))
            .set_type("unix")
            .set_name("stage")
            .set_parent_source("/")
            .build()
        )


@pytest.mark.parametrize(
    "builder,transformation",
    [
        pytest.param(
            ee.create_input_builder(),
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
            ee.create_output_builder(),
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
            ee.create_input_builder(),
            CopyTransformation(pathlib.Path("foo")),
        ),
        pytest.param(
            ee.create_output_builder(),
            CopyTransformation(pathlib.Path("foo")),
        ),
    ],
)
def test_io_direction_constraints(builder, transformation):
    builder.set_transformation(transformation)
