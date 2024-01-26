import dataclasses
from argparse import Namespace
from unittest.mock import MagicMock
from uuid import UUID

import pytest

from ert.cli import model_factory
from ert.libres_facade import LibresFacade
from ert.run_models import (
    EnsembleExperiment,
    EnsembleSmoother,
    IteratedEnsembleSmoother,
    MultipleDataAssimilation,
    SingleTestRun,
)


@pytest.mark.parametrize(
    "target_case, format_mode, expected",
    [
        ("test", False, "test"),
        (None, False, "default_smoother_update"),
        (None, True, "default_%d"),
    ],
)
def test_target_case_name(target_case, expected, format_mode, poly_case):
    args = Namespace(random_seed=None, current_case="default", target_case=target_case)
    assert (
        model_factory._target_case_name(poly_case, args, format_mode=format_mode)
        == expected
    )


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
    active_mask[0] = True
    active_mask[1] = True
    active_mask[2] = True
    active_mask[3] = True
    active_mask[4] = True
    active_mask[7] = True
    active_mask[8] = True
    assert model_factory._realizations(args, ensemble_size).tolist() == active_mask


def test_setup_single_test_run(poly_case, storage):
    model = model_factory._setup_single_test_run(
        poly_case,
        storage,
        Namespace(current_case="default", target_case=None, random_seed=None),
        UUID(int=0),
    )
    assert isinstance(model, SingleTestRun)
    assert model.id == UUID(int=0)
    assert model.simulation_arguments.current_case == "default"
    assert model.simulation_arguments.target_case is None
    assert model._storage == storage
    assert model.ert_config == poly_case


def test_setup_ensemble_experiment(poly_case, storage):
    args = Namespace(
        realizations=None,
        iter_num=1,
        current_case="default",
        target_case=None,
    )
    model = model_factory._setup_ensemble_experiment(
        poly_case,
        storage,
        args,
        UUID(int=0),
    )
    assert isinstance(model, EnsembleExperiment)

    sim_args_as_dict = dataclasses.asdict(model._simulation_arguments)
    assert "active_realizations" in sim_args_as_dict


def test_setup_ensemble_smoother(poly_case, storage):
    args = Namespace(
        realizations="0-4,7,8",
        current_case="default",
        target_case="test_case",
    )

    model = model_factory._setup_ensemble_smoother(
        poly_case, storage, args, UUID(int=0), MagicMock()
    )
    assert isinstance(model, EnsembleSmoother)
    assert model.simulation_arguments.current_case == "default"
    assert model.simulation_arguments.target_case == "test_case"
    assert (
        model.simulation_arguments.active_realizations
        == [True] * 5 + [False] * 2 + [True] * 2 + [False] * 91
    )


def test_setup_multiple_data_assimilation(poly_case, storage):
    args = Namespace(
        realizations="0-4,8",
        weights="6,4,2",
        target_case="test_case_%d",
        restart_run=False,
        prior_ensemble="default",
    )

    model = model_factory._setup_multiple_data_assimilation(
        poly_case, storage, args, UUID(int=0), MagicMock()
    )
    assert isinstance(model, MultipleDataAssimilation)
    assert model.simulation_arguments.weights == "6,4,2"
    assert (
        model.simulation_arguments.active_realizations
        == [True] * 5 + [False] * 3 + [True] * 1 + [False] * 91
    )
    assert model.simulation_arguments.target_case == "test_case_%d"
    assert model.simulation_arguments.prior_ensemble == "default"
    assert model.simulation_arguments.restart_run == False


def test_setup_iterative_ensemble_smoother(poly_case, storage):
    args = Namespace(
        realizations="0-4,7,8",
        current_case="default",
        target_case="test_case_%d",
        num_iterations="10",
    )

    model = model_factory._setup_iterative_ensemble_smoother(
        poly_case, storage, args, UUID(int=0), MagicMock()
    )
    assert isinstance(model, IteratedEnsembleSmoother)
    assert model.simulation_arguments.current_case == "default"
    assert model.simulation_arguments.target_case == "test_case_%d"
    assert (
        model.simulation_arguments.active_realizations
        == [True] * 5 + [False] * 2 + [True] * 2 + [False] * 91
    )
    assert model.simulation_arguments.num_iterations == 10
    assert poly_case.analysis_config.num_iterations == 10
