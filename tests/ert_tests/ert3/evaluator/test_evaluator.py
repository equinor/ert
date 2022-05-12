import asyncio

import ert
import ert3
import pytest
from ert_shared.async_utils import get_event_loop


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
    workspace,
    config,
    base_ensemble_dict,
    coeffs,
    expected,
    request,
    has_unix_config,
    plugin_registry,
):
    stages_config = request.getfixturevalue(config)
    storage_path = workspace._path / ".ert" / "tmp" / "test"
    input_transmitters = get_inputs(coeffs)
    base_ensemble_dict["size"] = len(coeffs)
    base_ensemble_dict["storage_type"] = "shared_disk"
    ensemble_config = ert3.config.load_ensemble_config(
        base_ensemble_dict, plugin_registry=plugin_registry
    )

    experiment_run_config = ert3.config.ExperimentRunConfig(
        stages_config,
        ensemble_config,
        ert3.config.ParametersConfig.parse_obj([]),
    )
    stage = experiment_run_config.get_stage()

    step_builder = (
        ert.ensemble_evaluator.StepBuilder()
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
            stage,
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
    "active_mask",
    [([True, True]), ([True, False]), ([False, True]), ([False, False])],
)
def test_inactive_realizations(
    active_mask,
    workspace,
    base_ensemble_dict,
    function_stages_config,
    plugin_registry,
):
    coeffs = [(0, 0, 0), (1, 1, 1)]
    storage_path = workspace._path / ".ert" / "tmp" / "test"
    base_ensemble_dict["size"] = len(coeffs)
    base_ensemble_dict["storage_type"] = "shared_disk"
    input_transmitters = get_inputs(coeffs)
    ensemble_config = ert3.config.load_ensemble_config(
        base_ensemble_dict, plugin_registry=plugin_registry
    )

    experiment_run_config = ert3.config.ExperimentRunConfig(
        function_stages_config,
        ensemble_config,
        ert3.config.ParametersConfig.parse_obj([]),
    )
    stage = experiment_run_config.get_stage()
    step_builder = (
        ert.ensemble_evaluator.StepBuilder()
        .set_name("dummy-function-only_step")
        .set_type("function")
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

    ensemble = ert3.evaluator.build_ensemble(
        stage,
        ensemble_config.forward_model.driver,
        ensemble_config.size,
        step_builder,
        active_mask,
    )

    evaluation_records = ert3.evaluator.evaluate(ensemble, range(1024, 65535))

    for realization_index, active in enumerate(active_mask):
        if active is False:
            with pytest.raises(RuntimeError, match="cannot load untransmitted record"):
                asyncio.get_event_loop().run_until_complete(
                    evaluation_records[realization_index]["polynomial_output"].load()
                )
        else:
            assert (
                asyncio.get_event_loop()
                .run_until_complete(
                    evaluation_records[realization_index]["polynomial_output"].load()
                )
                .data
            )


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
def test_inline_script(commandline, parsed_name, parsed_args, plugin_registry):
    """Verify that the ensemble builder will obey quotations in order
    to support inlined shell scripts"""

    # ert3.config.Unix contains plugged-in configurations, so load those first.
    ert3.config.create_stages_config(plugin_registry=plugin_registry)

    step = ert3.config.Unix(
        name="step with inlined script",
        input=[],
        script=tuple([commandline]),
        output=[],
        transportable_commands=tuple(),
    )
    step_builder = (
        ert.ensemble_evaluator.StepBuilder()
        .set_name("inline_script_test")
        .set_type("unix")
    )

    ensemble = ert3.evaluator.build_ensemble(
        stage=step, driver="local", ensemble_size=1, step_builder=step_builder
    )

    job = ensemble.reals[0].steps[0].jobs[0]
    assert job.name == parsed_name
    assert job.args == tuple(parsed_args)
