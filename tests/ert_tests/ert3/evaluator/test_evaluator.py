import asyncio

import pytest

import ert
import ert3

TEST_PARAMETRIZATION = [
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
]


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
    asyncio.get_event_loop().run_until_complete(asyncio.gather(*futures))
    return input_records


@pytest.mark.requires_ert_storage
@pytest.mark.parametrize("coeffs, expected", TEST_PARAMETRIZATION)
def test_evaluator_script(
    workspace, stages_config, base_ensemble_dict, coeffs, expected
):
    storage_path = workspace / ".ert" / "tmp" / "test"
    input_records = get_inputs(coeffs)
    base_ensemble_dict["size"] = len(coeffs)
    base_ensemble_dict["storage_type"] = "shared_disk"
    ensemble = ert3.config.load_ensemble_config(base_ensemble_dict)

    evaluation_records = ert3.evaluator.evaluate(
        storage_path,
        input_records,
        ensemble,
        stages_config,
    )

    for iens, transmitter_map in evaluation_records.items():
        record = asyncio.get_event_loop().run_until_complete(
            transmitter_map["polynomial_output"].load()
        )
        transmitter_map["polynomial_output"] = record.data

    expected = {iens: {"polynomial_output": data} for iens, data in enumerate(expected)}
    assert expected == evaluation_records


@pytest.mark.requires_ert_storage
@pytest.mark.parametrize("coeffs, expected", TEST_PARAMETRIZATION)
def test_evaluator_function(
    workspace, function_stages_config, base_ensemble_dict, coeffs, expected
):
    storage_path = workspace / ".ert" / "tmp" / "test"

    input_records = get_inputs(coeffs)
    base_ensemble_dict["size"] = len(coeffs)
    base_ensemble_dict["storage_type"] = "shared_disk"

    ensemble = ert3.config.load_ensemble_config(base_ensemble_dict)

    evaluation_records = ert3.evaluator.evaluate(
        storage_path,
        input_records,
        ensemble,
        function_stages_config,
    )

    for iens, transmitter_map in evaluation_records.items():
        record = asyncio.get_event_loop().run_until_complete(
            transmitter_map["polynomial_output"].load()
        )
        transmitter_map["polynomial_output"] = record.data

    expected = {iens: {"polynomial_output": data} for iens, data in enumerate(expected)}
    assert expected == evaluation_records
