import math
import queue
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from ert.config import ErtConfig, ObservationSettings
from ert.run_models import MultipleDataAssimilation as mda
from ert.run_models import model_factory


@pytest.mark.parametrize(
    ("weights", "expected"),
    [
        ("2, 2, 2, 2", [4] * 4),
        ("1, 2, 4, ", [1.75, 3.5, 7.0]),
        ("1.414213562373095, 1.414213562373095", [2, 2]),
    ],
)
def test_that_parse_weights_returns_expected_values(weights, expected):
    weights = mda.parse_weights(weights)
    assert weights == expected
    assert math.isclose(np.reciprocal(weights).sum(), 1.0)


def test_that_non_numeric_weight_raises_value_error():
    with pytest.raises(ValueError, match="could not convert string to float: 'error'"):
        mda.parse_weights("2, error, 2, 2")


@pytest.mark.parametrize(
    "weights",
    [
        "2, -1, 2, 2",
        "2.0, 0.0, 2, 2",
        "-1, -1, -1, 0",
        "0, 0, 0, 0",
        "0.0,1.0, 2.0, 3.0",
    ],
)
def test_that_zero_or_negative_weights_raise_value_error(weights):
    with pytest.raises(
        ValueError,
        match=f"Invalid weights: {weights}. Weights must be positive non zero numbers.",
    ):
        mda.parse_weights(weights)


@pytest.mark.filterwarnings("ignore:MIN_REALIZATIONS")
def test_that_runpaths_of_failed_realizations_survive_intermediate_deletion(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    model = model_factory._setup_multiple_data_assimilation(
        ErtConfig.from_file_contents(f"NUM_REALIZATIONS 2\nENSPATH {tmp_path}"),
        Namespace(
            realizations="0,1",
            weights="4,2,1",
            target_ensemble="iter-%d",
            prior_ensemble_id=None,
            experiment_name="es-mda",
            delete_intermediate_runpaths=True,
        ),
        ObservationSettings(),
        queue.SimpleQueue(),
    )

    intermediate_iteration = 1
    succeeded, failed = 0, 1
    runpaths = {
        realization: Path(
            model._run_paths.get_paths([realization], intermediate_iteration)[0]
        )
        for realization in (succeeded, failed)
    }
    for runpath in runpaths.values():
        runpath.mkdir(parents=True)

    # Evaluation deactivates the realizations whose forward models failed
    model.active_realizations = [True, False]

    model._delete_intermediate_runpaths(MagicMock(iteration=intermediate_iteration))

    assert not runpaths[succeeded].exists()
    assert runpaths[failed].exists(), (
        "Runpath of the failed realization was deleted, destroying the only "
        "record of why it failed; its responses never reached storage"
    )


@pytest.mark.filterwarnings("ignore:MIN_REALIZATIONS")
@pytest.mark.parametrize(
    ("delete_intermediate_runpaths", "posterior_iteration", "expect_deleted"),
    [
        pytest.param(True, 1, True, id="first_intermediate_iteration_is_deleted"),
        pytest.param(True, 2, True, id="second_intermediate_iteration_is_deleted"),
        pytest.param(True, 3, False, id="last_iteration_is_kept"),
        pytest.param(False, 1, False, id="nothing_is_deleted_when_flag_is_unset"),
        pytest.param(False, 2, False, id="nothing_is_deleted_when_flag_is_unset"),
    ],
)
def test_that_only_intermediate_runpaths_are_deleted_when_keeping_first_and_last(
    tmp_path,
    monkeypatch,
    delete_intermediate_runpaths,
    posterior_iteration,
    expect_deleted,
):
    monkeypatch.chdir(tmp_path)
    model = model_factory._setup_multiple_data_assimilation(
        ErtConfig.from_file_contents(f"NUM_REALIZATIONS 2\nENSPATH {tmp_path}"),
        Namespace(
            realizations="0,1",
            weights="4,2,1",
            target_ensemble="iter-%d",
            prior_ensemble_id=None,
            experiment_name="es-mda",
            delete_intermediate_runpaths=delete_intermediate_runpaths,
        ),
        ObservationSettings(),
        queue.SimpleQueue(),
    )

    all_iterations = range(len(model._parsed_weights) + 1)
    paths_per_iteration = {
        iteration: [
            Path(path) for path in model._run_paths.get_paths([0, 1], iteration)
        ]
        for iteration in all_iterations
    }
    for paths in paths_per_iteration.values():
        for path in paths:
            path.mkdir(parents=True)

    model._delete_intermediate_runpaths(MagicMock(iteration=posterior_iteration))

    for iteration, paths in paths_per_iteration.items():
        should_be_gone = expect_deleted and iteration == posterior_iteration
        for path in paths:
            assert path.exists() is not should_be_gone, (
                f"Runpath {path} for iteration {iteration} was "
                f"{'kept' if path.exists() else 'deleted'}, expected the opposite"
            )
