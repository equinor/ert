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
    "eval_type, coeffs, expected",
    [
        ("function_evaluation", [(0, 0, 0)], [[0] * 10]),
        ("evaluation", [(0, 0, 0)], [[0] * 10]),
        (
            "function_evaluation",
            [(1.5, 2.5, 3.5)],
            [[3.5, 7.5, 14.5, 24.5, 37.5, 53.5, 72.5, 94.5, 119.5, 147.5]],
        ),
        (
            "evaluation",
            [(1.5, 2.5, 3.5)],
            [[3.5, 7.5, 14.5, 24.5, 37.5, 53.5, 72.5, 94.5, 119.5, 147.5]],
        ),
        (
            "function_evaluation",
            [(1.5, 2.5, 3.5), (5, 4, 3)],
            [
                [3.5, 7.5, 14.5, 24.5, 37.5, 53.5, 72.5, 94.5, 119.5, 147.5],
                [3, 12, 31, 60, 99, 148, 207, 276, 355, 444],
            ],
        ),
        (
            "evaluation",
            [(1.5, 2.5, 3.5), (5, 4, 3)],
            [
                [3.5, 7.5, 14.5, 24.5, 37.5, 53.5, 72.5, 94.5, 119.5, 147.5],
                [3, 12, 31, 60, 99, 148, 207, 276, 355, 444],
            ],
        ),
    ],
)
def test_evaluator(eval_type, coeffs, expected, tmpdir):
    with tmpdir.as_cwd():
        shutil.copytree(_EXAMPLES_ROOT / "polynomial", "polynomial")
        workspace_root = tmpdir / "polynomial"
        workspace_root.chdir()

        input_records = {}
        input_records["coefficients"] = ert3.data.EnsembleRecord(
            records=[
                ert3.data.Record(data={"a": a, "b": b, "c": c}) for (a, b, c) in coeffs
            ]
        )
        input_records = ert3.data.MultiEnsembleRecord(ensemble_records=input_records)
        ensemble_config = (
            workspace_root
            / ert3.workspace.EXPERIMENTS_BASE
            / eval_type
            / "ensemble.yml"
        )
        with open(ensemble_config) as f:
            raw_ensemble_config = yaml.safe_load(f)
            raw_ensemble_config["size"] = len(coeffs)
            ensemble_config = ert3.config.load_ensemble_config(raw_ensemble_config)
        stages_config = ert3.console._console._load_stages_config(workspace_root)

        evaluation_responses = ert3.evaluator.evaluate(
            workspace_root,
            "test_evaluation",
            input_records,
            ensemble_config,
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
