from unittest.mock import MagicMock

import pytest

from ert.run_models import EnsembleExperiment


@pytest.mark.parametrize(
    "active_mask, expected",
    [
        ([True, True, True, True], True),
        ([False, False, True, False], False),
        ([], False),
        ([False, True, True], True),
        ([False, False, True], False),
    ],
)
def test_check_if_runpath_exists(
    create_dummy_run_path,
    active_mask: list,
    expected: bool,
):
    def get_run_path_mock(realizations, iteration=None):
        if iteration is not None:
            return [f"out/realization-{r}/iter-{iteration}" for r in realizations]
        return [f"out/realization-{r}" for r in realizations]

    EnsembleExperiment.validate = MagicMock()
    ensemble_experiment = EnsembleExperiment(
        *[MagicMock()] * 2 + [active_mask, MagicMock(), None] + [MagicMock()] * 4
    )
    ensemble_experiment.run_paths.get_paths = get_run_path_mock
    assert ensemble_experiment.check_if_runpath_exists() == expected
