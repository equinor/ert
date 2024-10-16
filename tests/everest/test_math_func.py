import itertools
import os

import numpy as np
import pandas as pd
import pytest

from everest import ConfigKeys as CK
from everest.config import EverestConfig
from everest.config.export_config import ExportConfig
from everest.export import export
from everest.suite import _EverestWorkflow
from everest.util import makedirs_if_needed

CONFIG_FILE_MULTIOBJ = "config_multiobj.yml"
CONFIG_FILE_ADVANCED = "config_advanced.yml"
CONFIG_AUTO_SCALED_CONTROLS = "config_auto_scaled_controls.yml"
CONFIG_FILE_REMOVE_RUN_PATH = "config_remove_run_path.yml"


@pytest.mark.integration_test
def test_math_func_multiobj(copy_math_func_test_data_to_tmp):
    config = EverestConfig.load_file(CONFIG_FILE_MULTIOBJ)

    workflow = _EverestWorkflow(config)
    assert workflow is not None
    workflow.start_optimization()

    # Check resulting points
    x, y, z = (workflow.result.controls["point_" + p] for p in ("x", "y", "z"))
    assert x == pytest.approx(0.0, abs=0.05)
    assert y == pytest.approx(0.0, abs=0.05)
    assert z == pytest.approx(0.5, abs=0.05)

    # Check the optimum values for each object.
    optim_p = workflow.result.expected_objectives["distance_p"]
    optim_q = workflow.result.expected_objectives["distance_q"]
    assert optim_p == pytest.approx(-0.5, abs=0.05)
    assert optim_q == pytest.approx(-4.5, abs=0.05)

    # The overall optimum is a weighted average of the objectives
    assert workflow.result.total_objective == pytest.approx(
        (-0.5 * (2.0 / 3.0) * 1.5) + (-4.5 * (1.0 / 3.0) * 1.0), abs=0.01
    )

    # Test conversion to pandas DataFrame
    if config.export is None:
        config.export = ExportConfig(discard_rejected=False)

    df = export(config)
    ok_evals = df[(df["is_gradient"] == 0) & (df["success"] == 1)]

    # Three points in this case are increasing the merit
    assert len(ok_evals[ok_evals["increased_merit"] == 1]) == 2

    first = ok_evals.iloc[0]
    best = ok_evals.iloc[-1]
    assert first["point_x"] == 0
    assert first["point_y"] == 0
    assert first["point_z"] == 0
    assert first["distance_p"] == -(0.5 * 0.5 * 3)
    assert first["distance_q"] == -(1.5 * 1.5 * 2 + 0.5 * 0.5)
    assert first["sim_avg_obj"] == (-0.75 * (2.0 / 3.0) * 1.5) + (
        -4.75 * (1.0 / 3.0) * 1.0
    )

    assert best["point_x"] == pytest.approx(x)
    assert best["point_y"] == pytest.approx(y)
    assert best["point_z"] == pytest.approx(z)
    assert best["distance_p"] == pytest.approx(optim_p)
    assert best["distance_q"] == pytest.approx(optim_q)
    assert best["sim_avg_obj"] == pytest.approx(workflow.result.total_objective)

    test_space = itertools.product(
        (first, best),
        (
            ("distance_p", 2.0 / 3, 1.5),
            ("distance_q", 1.0 / 3, 1),
        ),
    )
    for row, (obj_name, weight, norm) in test_space:
        assert row[obj_name] * norm == row[obj_name + "_norm"]
        assert row[obj_name] * weight * norm == pytest.approx(
            row[obj_name + "_weighted_norm"]
        )

    assert first["realization_weight"] == 1.0
    assert best["realization_weight"] == 1.0

    # check exported sim_avg_obj against dakota_tabular
    dt = pd.read_csv(
        os.path.join(config.optimization_output_dir, "dakota", "dakota_tabular.dat"),
        sep=" +",
        engine="python",
    )
    dt.sort_values(by=["%eval_id"], inplace=True)
    ok_evals = ok_evals.sort_values(by=["batch"])
    for a, b in zip(
        dt["obj_fn"],  # pylint: disable=unsubscriptable-object
        ok_evals["sim_avg_obj"],
    ):
        # Opposite, because ropt negates values before passing to dakota
        assert -a == pytest.approx(b)


@pytest.mark.integration_test
def test_math_func_advanced(copy_math_func_test_data_to_tmp):
    config = EverestConfig.load_file(CONFIG_FILE_ADVANCED)

    workflow = _EverestWorkflow(config)
    assert workflow is not None
    workflow.start_optimization()

    point_names = ["x-0", "x-1", "x-2"]
    # Check resulting points
    x0, x1, x2 = (workflow.result.controls["point_" + p] for p in point_names)
    assert x0 == pytest.approx(0.1, abs=0.05)
    assert x1 == pytest.approx(0.0, abs=0.05)
    assert x2 == pytest.approx(0.4, abs=0.05)

    # Check optimum value
    assert pytest.approx(workflow.result.total_objective, abs=0.1) == -(
        0.25 * (1.6**2 + 1.5**2 + 0.1**2) + 0.75 * (0.4**2 + 0.5**2 + 0.1**2)
    )
    # Expected distance is the weighted average of the (squared) distances
    #  from (x, y, z) to (-1.5, -1.5, 0.5) and (0.5, 0.5, 0.5)
    w = config.model.realizations_weights
    assert w == [0.25, 0.75]
    dist_0 = (x0 + 1.5) ** 2 + (x1 + 1.5) ** 2 + (x2 - 0.5) ** 2
    dist_1 = (x0 - 0.5) ** 2 + (x1 - 0.5) ** 2 + (x2 - 0.5) ** 2
    expected_opt = -(w[0] * (dist_0) + w[1] * (dist_1))
    assert expected_opt == pytest.approx(workflow.result.total_objective, abs=0.001)

    # Test conversion to pandas DataFrame
    df = export(config)
    ok_evals = df[(df["is_gradient"] == 0) & (df["success"] == 1)]

    ok_evals_0 = ok_evals[ok_evals["realization"] == 0]
    best_0 = ok_evals_0.iloc[-1]
    assert best_0["point_{}".format(point_names[0])] == pytest.approx(x0)
    assert best_0["point_{}".format(point_names[1])] == pytest.approx(x1)
    assert best_0["point_{}".format(point_names[2])] == pytest.approx(x2)
    assert best_0["distance"] == pytest.approx(-dist_0, abs=0.001)
    assert best_0["real_avg_obj"] == pytest.approx(
        workflow.result.total_objective, abs=0.001
    )
    assert best_0["realization_weight"] == 0.25

    ok_evals_1 = ok_evals[ok_evals["realization"] == 2]
    best_1 = ok_evals_1.iloc[-1]
    assert best_1["point_{}".format(point_names[0])] == pytest.approx(x0)
    assert best_1["point_{}".format(point_names[1])] == pytest.approx(x1)
    assert best_1["point_{}".format(point_names[2])] == pytest.approx(x2)
    assert best_1["distance"] == pytest.approx(-dist_1, abs=0.001)
    assert best_1["real_avg_obj"] == pytest.approx(
        workflow.result.total_objective, abs=0.001
    )
    assert best_1["realization_weight"] == 0.75

    # check functionality of export batch filtering
    if CK.EXPORT not in config:
        config.export = ExportConfig()

    exp_nunique = 2
    batches_list = [0, 2]
    config.export.batches = batches_list

    batch_filtered_df = export(config)
    n_unique_batches = batch_filtered_df["batch"].nunique()
    unique_batches = np.sort(batch_filtered_df["batch"].unique()).tolist()

    assert exp_nunique == n_unique_batches
    assert batches_list == unique_batches


@pytest.mark.integration_test
def test_remove_run_path(copy_math_func_test_data_to_tmp):
    config = EverestConfig.load_file(CONFIG_FILE_REMOVE_RUN_PATH)

    simulation_should_fail = "simulation_2"
    # Add to the config dictionary what simulation needs to fail
    config.forward_model[config.forward_model.index("toggle_failure")] = (
        "toggle_failure --fail %s" % simulation_should_fail
    )

    simulation_dir = config.simulation_dir

    _EverestWorkflow(config).start_optimization()

    # Check the failed simulation folder still exists
    assert os.path.exists(
        os.path.join(simulation_dir, "batch_0/geo_realization_0/simulation_2")
    ), "Simulation folder should be there, something went wrong and was removed!"

    # Check the successful simulation folders do not exist
    assert not os.path.exists(
        os.path.join(simulation_dir, "batch_0/geo_realization_0/simulation_0")
    ), "Simulation folder should not be there, something went wrong!"

    assert not os.path.exists(
        os.path.join(simulation_dir, "batch_0/geo_realization_0/simulation_1")
    ), "Simulation folder should not be there, something went wrong!"

    # Manually rolling the output folder between two runs
    makedirs_if_needed(config.output_dir, roll_if_exists=True)

    config.simulator = None
    _EverestWorkflow(config).start_optimization()

    # Check the all simulation folder exist when delete_run_path is set to False
    assert os.path.exists(
        os.path.join(simulation_dir, "batch_0/geo_realization_0/simulation_2")
    ), "Simulation folder should be there, something went wrong and was removed!"

    assert os.path.exists(
        os.path.join(simulation_dir, "batch_0/geo_realization_0/simulation_0")
    ), "Simulation folder should be there, something went wrong and was removed"

    assert os.path.exists(
        os.path.join(simulation_dir, "batch_0/geo_realization_0/simulation_1")
    ), "Simulation folder should be there, something went wrong and was removed"


def test_math_func_auto_scaled_controls(copy_math_func_test_data_to_tmp):
    config = EverestConfig.load_file(CONFIG_AUTO_SCALED_CONTROLS)

    workflow = _EverestWorkflow(config)
    assert workflow is not None
    workflow.start_optimization()

    # Check resulting points
    x, y, z = (workflow.result.controls["point_" + p] for p in ("x", "y", "z"))

    assert x == pytest.approx(0.25, abs=0.05)
    assert y == pytest.approx(0.25, abs=0.05)
    assert z == pytest.approx(0.5, abs=0.05)

    # Check optimum value
    optim = -workflow.result.total_objective  # distance is provided as -distance
    expected_dist = 0.25**2 + 0.25**2
    assert expected_dist == pytest.approx(optim, abs=0.05)
    assert expected_dist == pytest.approx(optim, abs=0.05)
