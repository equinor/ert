from argparse import Namespace
from unittest.mock import MagicMock
from uuid import uuid1

import pytest

import ert
from ert.config import ConfigValidationError, ErtConfig, ModelConfig
from ert.libres_facade import LibresFacade
from ert.run_models import (
    EnsembleExperiment,
    EnsembleSmoother,
    IteratedEnsembleSmoother,
    MultipleDataAssimilation,
    SingleTestRun,
    model_factory,
)
from ert.run_models.evaluate_ensemble import EvaluateEnsemble


@pytest.mark.parametrize(
    "target_ensemble, expected",
    [
        (None, "default_%d"),
    ],
)
def test_iterative_ensemble_format(target_ensemble, expected, poly_case):
    args = Namespace(
        random_seed=None, current_ensemble="default", target_ensemble=target_ensemble
    )
    assert model_factory._iterative_ensemble_format(args) == expected


def test_default_realizations(poly_case):
    facade = LibresFacade(poly_case)
    args = Namespace(realizations=None)
    assert (
        model_factory._realizations(args, facade.get_ensemble_size()).tolist()
        == [True] * facade.get_ensemble_size()
    )


def test_custom_realizations(poly_case):
    facade = LibresFacade(poly_case)
    args = Namespace(realizations="0-4,7,8")
    ensemble_size = facade.get_ensemble_size()
    active_mask = [False] * ensemble_size
    active_mask[0:5] = [True] * 5
    active_mask[7:9] = [True] * 2
    assert model_factory._realizations(args, ensemble_size).tolist() == active_mask


def test_setup_single_test_run(poly_case, storage):
    model = model_factory._setup_single_test_run(
        poly_case,
        storage,
        Namespace(
            current_ensemble="current-ensemble",
            target_ensemble=None,
            random_seed=None,
            experiment_name=None,
        ),
        MagicMock(),
    )
    assert isinstance(model, SingleTestRun)
    assert model._storage == storage
    assert model.ert_config == poly_case


def test_setup_single_test_run_with_ensemble(poly_case, storage):
    model = model_factory._setup_single_test_run(
        poly_case,
        storage,
        Namespace(
            current_ensemble="current-ensemble",
            target_ensemble=None,
            random_seed=None,
            experiment_name=None,
        ),
        MagicMock(),
    )
    assert isinstance(model, SingleTestRun)
    assert model._storage == storage
    assert model.ert_config == poly_case


def test_setup_ensemble_experiment(poly_case, storage):
    args = Namespace(
        realizations=None,
        iter_num=1,
        current_ensemble="default",
        target_ensemble=None,
        experiment_name="ensemble_experiment",
    )
    model = model_factory._setup_ensemble_experiment(
        poly_case,
        storage,
        args,
        MagicMock(),
    )
    assert isinstance(model, EnsembleExperiment)

    assert model.active_realizations == [True] * 100


def test_setup_ensemble_smoother(poly_case, storage):
    args = Namespace(
        realizations="0-4,7,8",
        current_ensemble="default",
        target_ensemble="test_case",
        experiment_name=None,
    )

    model = model_factory._setup_ensemble_smoother(
        poly_case, storage, args, MagicMock(), MagicMock()
    )
    assert isinstance(model, EnsembleSmoother)
    assert (
        model.active_realizations
        == [True] * 5 + [False] * 2 + [True] * 2 + [False] * 91
    )


def test_setup_multiple_data_assimilation(poly_case, storage):
    args = Namespace(
        realizations="0-4,8",
        weights="6,4,2",
        target_ensemble="test_case_%d",
        restart_run=False,
        prior_ensemble_id="b272fe09-83ac-4744-b667-9a0a5415420b",
        experiment_name="My-experiment",
        starting_iteration=0,
    )

    model = model_factory._setup_multiple_data_assimilation(
        poly_case, storage, args, MagicMock(), MagicMock()
    )
    assert isinstance(model, MultipleDataAssimilation)
    assert model.weights == MultipleDataAssimilation.parse_weights("6,4,2")
    assert (
        model.active_realizations
        == [True] * 5 + [False] * 3 + [True] * 1 + [False] * 91
    )
    assert model.target_ensemble_format == "test_case_%d"
    assert model.prior_ensemble_id == "b272fe09-83ac-4744-b667-9a0a5415420b"
    assert model.restart_run == False


def test_setup_iterative_ensemble_smoother(poly_case, storage):
    args = Namespace(
        realizations="0-4,7,8",
        target_ensemble="test_case_%d",
        num_iterations=10,
        experiment_name=None,
    )

    model = model_factory._setup_iterative_ensemble_smoother(
        poly_case, storage, args, MagicMock(), MagicMock()
    )
    assert isinstance(model, IteratedEnsembleSmoother)
    assert model.target_ensemble_format == "test_case_%d"
    assert (
        model.active_realizations
        == [True] * 5 + [False] * 2 + [True] * 2 + [False] * 91
    )
    assert model._total_iterations == 10


@pytest.mark.parametrize(
    "restart_from_iteration, expected_path",
    [
        [
            0,
            [
                "realization-0/iter-1",
                "realization-0/iter-2",
                "realization-0/iter-3",
                "realization-1/iter-1",
                "realization-1/iter-2",
                "realization-1/iter-3",
            ],
        ],
        [
            1,
            [
                "realization-0/iter-2",
                "realization-0/iter-3",
                "realization-1/iter-2",
                "realization-1/iter-3",
            ],
        ],
        [2, ["realization-0/iter-3", "realization-1/iter-3"]],
        [3, []],
    ],
)
def test_multiple_data_assimilation_restart_paths(
    tmp_path, monkeypatch, restart_from_iteration, expected_path
):
    monkeypatch.chdir(tmp_path)
    args = Namespace(
        realizations="0,1",
        weights="6,4,2",
        target_ensemble="restart_case_%d",
        restart_run=True,
        prior_ensemble_id=str(uuid1()),
        experiment_name=None,
    )
    monkeypatch.setattr(
        ert.run_models.base_run_model.BaseRunModel, "validate", MagicMock()
    )
    storage_mock = MagicMock()
    ensemble_mock = MagicMock()
    ensemble_mock.iteration = restart_from_iteration
    config = ErtConfig(model_config=ModelConfig(num_realizations=2))
    storage_mock.get_ensemble.return_value = ensemble_mock
    model = model_factory._setup_multiple_data_assimilation(
        config, storage_mock, args, MagicMock(), MagicMock()
    )
    base_path = tmp_path / "simulations"
    expected_path = [str(base_path / expected) for expected in expected_path]
    assert set(model.paths) == set(expected_path)


@pytest.mark.parametrize(
    "analysis_mode",
    [
        model_factory._setup_multiple_data_assimilation,
        model_factory._setup_ensemble_smoother,
        model_factory._setup_iterative_ensemble_smoother,
    ],
)
def test_num_realizations_specified_incorrectly_raises(analysis_mode):
    config = ErtConfig(model_config=ModelConfig(num_realizations=1))
    args = Namespace(
        realizations="0",
        weights="6,4,2",
        target_ensemble="restart_case_%d",
        restart_run=True,
        prior_ensemble_id=str(uuid1()),
        experiment_name=None,
    )

    with pytest.raises(
        ConfigValidationError,
        match="Number of active realizations must be at least 2 for an update step",
    ):
        analysis_mode(config, MagicMock(), args, MagicMock(), MagicMock())


@pytest.mark.parametrize(
    "ensemble_iteration, expected_path",
    [
        [0, ["realization-0/iter-0"]],
        [1, ["realization-0/iter-1"]],
        [2, ["realization-0/iter-2"]],
        [100, ["realization-0/iter-100"]],
    ],
)
def test_evaluate_ensemble_paths(
    tmp_path, monkeypatch, ensemble_iteration, expected_path
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        ert.run_models.base_run_model.BaseRunModel, "validate", MagicMock()
    )
    storage_mock = MagicMock()
    ensemble_mock = MagicMock()
    ensemble_mock.iteration = ensemble_iteration
    config = ErtConfig(model_config=ModelConfig(num_realizations=2))
    storage_mock.get_ensemble.return_value = ensemble_mock
    model = EvaluateEnsemble(
        [True], 1, str(uuid1(0)), 1234, config, storage_mock, MagicMock(), MagicMock()
    )
    base_path = tmp_path / "simulations"
    expected_path = [str(base_path / expected) for expected in expected_path]
    assert set(model.paths) == set(expected_path)
