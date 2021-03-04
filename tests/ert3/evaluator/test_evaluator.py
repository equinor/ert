import pytest
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
    input_records["coefficients"] = ert3.data.EnsembleRecord(
        records=[
            ert3.data.Record(data={"a": a, "b": b, "c": c}) for (a, b, c) in coeffs
        ]
    )
    return ert3.data.MultiEnsembleRecord(ensemble_records=input_records)


@pytest.mark.requires_ert_storage
@pytest.mark.parametrize("coeffs, expected", TEST_PARAMETRIZATION)
def test_evaluator_script(workspace, stages_config, ensemble, coeffs, expected):
    input_records = get_inputs(coeffs)
    ensemble.size = len(coeffs)

    evaluation_responses = ert3.evaluator.evaluate(
        workspace,
        "test_evaluation",
        input_records,
        ensemble,
        stages_config,
    )

    expected = ert3.data.MultiEnsembleRecord(
        ensemble_records={
            "polynomial_output": ert3.data.EnsembleRecord(
                records=[ert3.data.Record(data=poly_out) for poly_out in expected],
            )
        }
    )
    assert expected == evaluation_responses


@pytest.mark.requires_ert_storage
@pytest.mark.parametrize("coeffs, expected", TEST_PARAMETRIZATION)
def test_evaluator_function(
    workspace, function_stages_config, ensemble, coeffs, expected
):
    input_records = get_inputs(coeffs)
    ensemble.size = len(coeffs)

    evaluation_responses = ert3.evaluator.evaluate(
        workspace,
        "test_evaluation",
        input_records,
        ensemble,
        function_stages_config,
    )

    expected = ert3.data.MultiEnsembleRecord(
        ensemble_records={
            "polynomial_output": ert3.data.EnsembleRecord(
                records=[ert3.data.Record(data=poly_out) for poly_out in expected],
            )
        }
    )
    assert expected == evaluation_responses
