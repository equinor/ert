import os
import pathlib
import pytest
import shutil
import yaml

import ert3


_EXAMPLES_ROOT = (
    pathlib.Path(os.path.dirname(__file__)) / ".." / ".." / ".." / "examples"
)


@pytest.mark.parametrize(
    "coeffs, expected",
    [
        (
            [(0, 0, 0)],
            [{"polynomial_output": [0] * 10, "polynomial_function_output": [0] * 10}],
        ),
        (
            [(1.5, 2.5, 3.5)],
            [
                {
                    "polynomial_output": [
                        3.5,
                        7.5,
                        14.5,
                        24.5,
                        37.5,
                        53.5,
                        72.5,
                        94.5,
                        119.5,
                        147.5,
                    ],
                    "polynomial_function_output": [
                        3.5,
                        7.5,
                        14.5,
                        24.5,
                        37.5,
                        53.5,
                        72.5,
                        94.5,
                        119.5,
                        147.5,
                    ],
                }
            ],
        ),
        (
            [(1.5, 2.5, 3.5), (5, 4, 3)],
            [
                {
                    "polynomial_output": [
                        3.5,
                        7.5,
                        14.5,
                        24.5,
                        37.5,
                        53.5,
                        72.5,
                        94.5,
                        119.5,
                        147.5,
                    ],
                    "polynomial_function_output": [
                        3.5,
                        7.5,
                        14.5,
                        24.5,
                        37.5,
                        53.5,
                        72.5,
                        94.5,
                        119.5,
                        147.5,
                    ],
                },
                {
                    "polynomial_output": [3, 12, 31, 60, 99, 148, 207, 276, 355, 444],
                    "polynomial_function_output": [
                        3,
                        12,
                        31,
                        60,
                        99,
                        148,
                        207,
                        276,
                        355,
                        444,
                    ],
                },
            ],
        ),
    ],
)
def test_evaluator(coeffs, expected, tmpdir):
    with tmpdir.as_cwd():
        shutil.copytree(_EXAMPLES_ROOT / "polynomial", "polynomial")
        workspace_root = tmpdir / "polynomial"
        workspace_root.chdir()

        input_records = [
            {"coefficients": {"a": a, "b": b, "c": c}} for (a, b, c) in coeffs
        ]
        with open(workspace_root / "evaluation" / "ensemble.yml") as f:
            raw_ensemble_config = yaml.safe_load(f)
            raw_ensemble_config["size"] = len(input_records)
            ensemble_config = ert3.config.load_ensemble_config(raw_ensemble_config)
        with open(workspace_root / "stages.yml") as f:
            stages_config = ert3.config.load_stages_config(yaml.safe_load(f))

        evaluation_result = ert3.evaluator.evaluate(
            workspace_root,
            "test_evaluation",
            input_records,
            ensemble_config,
            stages_config,
        )
        for index, result in enumerate(evaluation_result):
            for record in result:
                assert result[record] == expected[index][record]
