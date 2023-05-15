from argparse import Namespace
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
    ert = poly_case
    args = Namespace(current_case="default", target_case=target_case)
    assert (
        model_factory._target_case_name(ert, args, format_mode=format_mode) == expected
    )


def test_default_realizations(poly_case):
    facade = LibresFacade(poly_case)
    args = Namespace(realizations=None)
    assert (
        model_factory._realizations(args, facade.get_ensemble_size())
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
    assert model_factory._realizations(args, ensemble_size) == active_mask


def test_setup_single_test_run(poly_case, storage):
    ert = poly_case

    model = model_factory._setup_single_test_run(
        ert, storage, Namespace(current_case="default"), UUID(int=0)
    )
    assert isinstance(model, SingleTestRun)
    assert len(model._simulation_arguments.keys()) == 2
    assert "active_realizations" in model._simulation_arguments


def test_setup_ensemble_experiment(poly_case, storage):
    ert = poly_case
    args = Namespace(realizations=None, iter_num=1, current_case="default")
    model = model_factory._setup_ensemble_experiment(
        ert,
        storage,
        args,
        UUID(int=0),
    )
    assert isinstance(model, EnsembleExperiment)
    assert len(model._simulation_arguments.keys()) == 3
    assert "active_realizations" in model._simulation_arguments


def test_setup_ensemble_smoother(poly_case, storage):
    ert = poly_case

    args = Namespace(
        realizations="0-4,7,8", current_case="default", target_case="test_case"
    )

    model = model_factory._setup_ensemble_smoother(
        ert,
        storage,
        args,
        UUID(int=0),
    )
    assert isinstance(model, EnsembleSmoother)
    assert len(model._simulation_arguments.keys()) == 4
    assert "active_realizations" in model._simulation_arguments
    assert "target_case" in model._simulation_arguments
    assert "analysis_module" in model._simulation_arguments


def test_setup_multiple_data_assimilation(poly_case, storage):
    ert = poly_case
    args = Namespace(
        realizations="0-4,7,8",
        weights="6,4,2",
        current_case="default",
        target_case="test_case_%d",
        start_iteration="0",
        restart_run=False,
        prior_ensemble="default",
    )

    model = model_factory._setup_multiple_data_assimilation(
        ert,
        storage,
        args,
        UUID(int=0),
    )
    assert isinstance(model, MultipleDataAssimilation)
    assert len(model._simulation_arguments.keys()) == 7
    assert "active_realizations" in model._simulation_arguments
    assert "target_case" in model._simulation_arguments
    assert "analysis_module" in model._simulation_arguments
    assert "weights" in model._simulation_arguments


def test_setup_iterative_ensemble_smoother(poly_case, storage):
    ert = poly_case
    args = Namespace(
        realizations="0-4,7,8",
        current_case="default",
        target_case="test_case_%d",
        num_iterations="10",
    )

    model = model_factory._setup_iterative_ensemble_smoother(
        ert,
        storage,
        args,
        UUID(int=0),
    )
    assert isinstance(model, IteratedEnsembleSmoother)
    assert len(model._simulation_arguments.keys()) == 5
    assert "active_realizations" in model._simulation_arguments
    assert "target_case" in model._simulation_arguments
    assert "analysis_module" in model._simulation_arguments
    assert "num_iterations" in model._simulation_arguments
    assert LibresFacade(ert).get_number_of_iterations() == 10
