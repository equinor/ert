import asyncio

import pytest

import ert
import ert3
from ert_shared.async_utils import get_event_loop
from ert_shared.ensemble_evaluator.ensemble.builder import create_step_builder


def get_inputs(coeffs):
    input_records = {}
    futures = []
    for iens, (a, b, c) in enumerate(coeffs):
        record_name = "coefficients"
        t = ert.data.InMemoryRecordTransmitter(record_name)
        futures.append(
            t.transmit_record(ert.data.NumericalRecord(data={"a": a, "b": b, "c": c}))
        )
        input_records[iens] = {record_name: t}
    get_event_loop().run_until_complete(asyncio.gather(*futures))
    return input_records


@pytest.mark.requires_ert_storage
@pytest.mark.parametrize(
    "coeffs, expected",
    [
        ([(0, 0, 0)], [[0] * 10]),
        (
            [(1.5, 2.5, 3.5)],
            [[3.5, 7.5, 14.5, 24.5, 37.5, 53.5, 72.5, 94.5, 119.5, 147.5]],
        ),
        (
            [(1.5, 2.5, 3.5), (5, 4, 3)],
            [
                [3.5, 7.5, 14.5, 24.5, 37.5, 53.5, 72.5, 94.5, 119.5, 147.5],
                [3, 12, 31, 60, 99, 148, 207, 276, 355, 444],
            ],
        ),
    ],
)
@pytest.mark.parametrize(
    "config, has_unix_config",
    [["stages_config", True], ["function_stages_config", False]],
)
def test_evaluator(
    workspace, config, base_ensemble_dict, coeffs, expected, request, has_unix_config
):
    stages_config = request.getfixturevalue(config)
    storage_path = workspace._path / ".ert" / "tmp" / "test"
    input_transmitters = get_inputs(coeffs)
    base_ensemble_dict["size"] = len(coeffs)
    base_ensemble_dict["storage_type"] = "shared_disk"
    ensemble_config = ert3.config.load_ensemble_config(base_ensemble_dict)

    experiment_run_config = ert3.config.ExperimentRunConfig(
        ert3.config.ExperimentConfig(type="evaluation"),
        stages_config,
        ensemble_config,
        ert3.config.ParametersConfig.parse_obj([]),
    )
    stage = experiment_run_config.get_stage()

    step_builder = (
        create_step_builder()
        .set_name(f"{stage.name}-only_step")
        .set_type("function" if isinstance(stage, ert3.config.Function) else "unix")
    )

    inputs = experiment_run_config.get_linked_inputs()
    stochastic_inputs = tuple(inputs[ert3.config.SourceNS.stochastic].values())

    ert3.evaluator.add_step_inputs(stochastic_inputs, input_transmitters, step_builder)

    ert3.evaluator.add_step_outputs(
        ensemble_config.storage_type,
        stage,
        storage_path,
        ensemble_config.size,
        step_builder,
    )

    if has_unix_config:
        ert3.evaluator.add_commands(
            stage.transportable_commands,
            base_ensemble_dict["storage_type"],
            storage_path,
            step_builder,
        )

    ensemble = ert3.evaluator.build_ensemble(
        stage, ensemble_config.forward_model.driver, ensemble_config.size, step_builder
    )

    evaluation_records = ert3.evaluator.evaluate(ensemble, range(1024, 65535))

    for _, transmitter_map in evaluation_records.items():
        record = asyncio.get_event_loop().run_until_complete(
            transmitter_map["polynomial_output"].load()
        )
        transmitter_map["polynomial_output"] = record.data

    expected = {iens: {"polynomial_output": data} for iens, data in enumerate(expected)}
    assert expected == evaluation_records


@pytest.mark.parametrize(
    "commandline, parsed_name, parsed_args",
    [
        ("echo inline", "echo", ["inline"]),
        ("sh -c 'echo inline'", "sh", ["-c", "echo inline"]),
        ('bash -c "echo inline"', "bash", ["-c", "echo inline"]),
        ("bash -c echo inline", "bash", ["-c", "echo", "inline"]),  # (user error)
        (
            "sh -c " + '"' + "echo '[0, 1, 2, 3]' > some_numbers.json" + '"',
            "sh",
            ["-c", "echo '[0, 1, 2, 3]' > some_numbers.json"],
        ),
    ],
)
def test_inline_script(commandline, parsed_name, parsed_args):
    """Verify that the ensemble builder will obey quotations in order
    to support inlined shell scripts"""
    step = ert3.config.Unix(
        name="step with inlined script",
        input=[],
        script=tuple([commandline]),
        output=[],
        transportable_commands=tuple(),
    )
    step_builder = create_step_builder().set_name("inline_script_test").set_type("unix")

    ensemble = ert3.evaluator.build_ensemble(
        stage=step, driver="local", ensemble_size=1, step_builder=step_builder
    )

    job = ensemble.get_reals()[0].get_steps()[0].get_jobs()[0]
    assert job.get_name() == parsed_name
    assert job.get_args() == tuple(parsed_args)
