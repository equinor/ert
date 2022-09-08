from argparse import Namespace

import pytest

from ert._c_wrappers.enkf import EnKFMain
from ert.libres_facade import LibresFacade
from ert.shared.cli import model_factory
from ert.shared.models import (
    EnsembleExperiment,
    EnsembleSmoother,
    IteratedEnsembleSmoother,
    MultipleDataAssimilation,
    SingleTestRun,
)


@pytest.fixture
def poly_example(setup_case):
    res_config = setup_case("local/poly_example", "poly.ert")
    ert = EnKFMain(res_config)
    facade = LibresFacade(ert)
    return ert, facade


@pytest.mark.parametrize(
    "target_case, format_mode, expected",
    [
        ("test", False, "test"),
        (None, False, "default_smoother_update"),
        (None, True, "default_%d"),
    ],
)
def test_target_case_name(target_case, expected, format_mode, poly_example):
    ert, facade = poly_example
    args = Namespace(target_case=target_case)
    assert (
        model_factory._target_case_name(
            ert, args, facade.get_current_case_name(), format_mode=format_mode
        )
        == expected
    )


def test_default_realizations(poly_example):
    _, facade = poly_example
    args = Namespace(realizations=None)
    assert (
        model_factory._realizations(args, facade.get_ensemble_size())
        == [True] * facade.get_ensemble_size()
    )


def test_custom_realizations(poly_example):
    _, facade = poly_example
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


def test_init_iteration_number(poly_example):
    ert, facade = poly_example
    args = Namespace(iter_num=10, realizations=None)
    model = model_factory._setup_ensemble_experiment(
        ert, args, facade.get_ensemble_size(), "experiment_id"
    )
    run_context = model.create_context()
    assert model._simulation_arguments["iter_num"] == 10
    assert run_context.iteration == 10


def test_setup_single_test_run(poly_example):
    ert, _ = poly_example

    model = model_factory._setup_single_test_run(ert, "experiment_id")
    assert isinstance(model, SingleTestRun)
    assert len(model._simulation_arguments.keys()) == 1
    assert "active_realizations" in model._simulation_arguments

    model.create_context()


def test_setup_ensemble_experiment(poly_example):
    ert, facade = poly_example
    args = Namespace(realizations=None, iter_num=1)
    model = model_factory._setup_ensemble_experiment(
        ert, args, facade.get_ensemble_size(), "experiment_id"
    )
    assert isinstance(model, EnsembleExperiment)
    assert len(model._simulation_arguments.keys()) == 2
    assert "active_realizations" in model._simulation_arguments

    model.create_context()


def test_setup_ensemble_smoother(poly_example):
    ert, facade = poly_example

    args = Namespace(realizations="0-4,7,8", target_case="test_case")

    model = model_factory._setup_ensemble_smoother(
        ert,
        args,
        facade.get_ensemble_size(),
        facade.get_current_case_name(),
        "experiment_id",
    )
    assert isinstance(model, EnsembleSmoother)
    assert len(model._simulation_arguments.keys()) == 3
    assert "active_realizations" in model._simulation_arguments
    assert "target_case" in model._simulation_arguments
    assert "analysis_module" in model._simulation_arguments

    model.create_context()


def test_setup_multiple_data_assimilation(poly_example):
    ert, facade = poly_example
    args = Namespace(
        realizations="0-4,7,8",
        weights="6,4,2",
        target_case="test_case_%d",
        start_iteration="0",
    )

    model = model_factory._setup_multiple_data_assimilation(
        ert,
        args,
        facade.get_ensemble_size(),
        facade.get_current_case_name(),
        "experiment_id",
    )
    assert isinstance(model, MultipleDataAssimilation)
    assert len(model._simulation_arguments.keys()) == 5
    assert "active_realizations" in model._simulation_arguments
    assert "target_case" in model._simulation_arguments
    assert "analysis_module" in model._simulation_arguments
    assert "weights" in model._simulation_arguments
    assert "start_iteration" in model._simulation_arguments
    model.create_context(0)


def test_setup_iterative_ensemble_smoother(poly_example):
    ert, facade = poly_example
    args = Namespace(
        realizations="0-4,7,8",
        target_case="test_case_%d",
        num_iterations="10",
    )

    model = model_factory._setup_iterative_ensemble_smoother(
        ert,
        args,
        facade.get_ensemble_size(),
        facade.get_current_case_name(),
        "experiment_id",
    )
    assert isinstance(model, IteratedEnsembleSmoother)
    assert len(model._simulation_arguments.keys()) == 4
    assert "active_realizations" in model._simulation_arguments
    assert "target_case" in model._simulation_arguments
    assert "analysis_module" in model._simulation_arguments
    assert "num_iterations" in model._simulation_arguments
    assert facade.get_number_of_iterations() == 10

    model.create_context(0)
