import logging
from datetime import datetime
from pathlib import Path
from textwrap import dedent

import numpy as np
import polars as pl
import pytest
from resdata.summary import Summary

from ert.config import ErtConfig
from ert.run_models._create_run_path import create_run_path
from ert.runpaths import Runpaths
from ert.storage import open_storage
from ert.storage.local_ensemble import (
    RealizationStorageState,
    load_parameters_and_responses_from_runpath,
)


@pytest.fixture()
def setup_case(storage, use_tmpdir, run_args):
    def func(config_text):
        Path("config.ert").write_text(config_text, encoding="utf-8")

        ert_config = ErtConfig.from_file("config.ert")
        prior_ensemble = storage.create_ensemble(
            storage.create_experiment(
                responses=ert_config.ensemble_config.response_configuration
            ),
            name="prior",
            ensemble_size=ert_config.runpath_config.num_realizations,
        )
        create_run_path(
            run_args=run_args(ert_config, prior_ensemble),
            ensemble=prior_ensemble,
            user_config_file=ert_config.user_config_file,
            env_vars=ert_config.env_vars,
            env_pr_fm_step=ert_config.env_pr_fm_step,
            forward_model_steps=ert_config.forward_model_steps,
            substitutions=ert_config.substitutions,
            parameters_file="parameters",
            runpaths=Runpaths.from_config(ert_config),
        )
        return prior_ensemble

    yield func


def run_simulator(time_step_count, start_date) -> Summary:
    summary = Summary.writer("SNAKE_OIL_FIELD", start_date, 10, 10, 10)

    summary.add_variable("FOPR", unit="SM3/DAY")
    summary.add_variable("FOPRH", unit="SM3/DAY")

    summary.add_variable("WOPR", wgname="OP1", unit="SM3/DAY")
    summary.add_variable("WOPRH", wgname="OP1", unit="SM3/DAY")

    mini_step_count = 10
    for report_step in range(time_step_count):
        for mini_step in range(mini_step_count):
            t_step = summary.add_t_step(
                report_step + 1, sim_days=report_step * mini_step_count + mini_step
            )
            t_step["FOPR"] = 1
            t_step["WOPR:OP1"] = 2
            t_step["FOPRH"] = 3
            t_step["WOPRH:OP1"] = 4

    return summary


@pytest.mark.usefixtures("copy_snake_oil_case_storage")
def test_load_forward_model(snake_oil_default_storage):
    """
    Checking that we are able to load from forward model
    """
    realisation_number = 0

    with open_storage("storage/snake_oil/ensemble", mode="w") as storage:
        # 'load_from_forward_model' requires the ensemble to be writeable...
        experiment = storage.get_experiment_by_name("ensemble-experiment")
        default = experiment.get_ensemble_by_name("default_0")

        loaded = load_parameters_and_responses_from_runpath(
            "storage/snake_oil/runpath/realization-<IENS>/iter-<ITER>",
            default,
            [realisation_number],
        )
        assert loaded == 1
        assert default.get_realization_mask_with_responses()[
            realisation_number
        ]  # Check that status is as expected


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.parametrize(
    "summary_configuration, expected",
    [
        pytest.param(
            None,
            (1, None),
            id=(
                "Checking that we are able to successfully load "
                "from forward model with eclipse case even though "
                "there is no eclipse file in the run path. This is "
                "because no SUMMARY is added to the config"
            ),
        ),
        pytest.param(
            "SUMMARY *",
            (0, "Could not find any summary file"),
            id=(
                "Check that loading fails if we have configured"
                "SUMMARY but no summary is available in the run path"
            ),
        ),
    ],
)
@pytest.mark.filterwarnings("ignore:Config contains a SUMMARY key")
def test_load_forward_model_summary(
    summary_configuration, storage, expected, caplog, run_args
):
    config_text = dedent(
        """
        NUM_REALIZATIONS 1
        ECLBASE SNAKE_OIL_FIELD
        REFCASE SNAKE_OIL_FIELD
        """
    )
    if summary_configuration:
        config_text += summary_configuration
    Path("config.ert").write_text(config_text, encoding="utf-8")
    # Create refcase
    ecl_sum = run_simulator(1, datetime(2014, 9, 10))
    ecl_sum.fwrite()

    ert_config = ErtConfig.from_file("config.ert")
    experiment_id = storage.create_experiment(
        responses=ert_config.ensemble_config.response_configuration
    )
    prior_ensemble = storage.create_ensemble(
        experiment_id, name="prior", ensemble_size=100
    )

    create_run_path(
        run_args=run_args(ert_config, prior_ensemble),
        ensemble=prior_ensemble,
        user_config_file=ert_config.user_config_file,
        env_vars=ert_config.env_vars,
        env_pr_fm_step=ert_config.env_pr_fm_step,
        forward_model_steps=ert_config.forward_model_steps,
        substitutions=ert_config.substitutions,
        parameters_file="parameters",
        runpaths=Runpaths.from_config(ert_config),
    )
    with caplog.at_level(logging.ERROR):
        loaded = load_parameters_and_responses_from_runpath(
            ert_config.runpath_config.runpath_format_string, prior_ensemble, [0]
        )
    expected_loaded, expected_log_message = expected
    assert loaded == expected_loaded
    if expected_log_message:
        assert expected_log_message in "".join(caplog.messages)


def test_load_forward_model_gen_data(setup_case):
    config_text = dedent(
        """
    NUM_REALIZATIONS 1
    GEN_DATA RESPONSE RESULT_FILE:response_%d.out REPORT_STEPS:0,1
        """
    )

    prior_ensemble = setup_case(config_text)
    run_path = Path("simulations/realization-0/iter-0/")
    (run_path / "response_0.out").write_text(
        "\n".join(["1", "2", "3"]), encoding="utf-8"
    )
    (run_path / "response_1.out").write_text(
        "\n".join(["4", "5", "5"]), encoding="utf-8"
    )
    (run_path / "response_0.out_active").write_text(
        "\n".join(["1", "0", "1"]), encoding="utf-8"
    )

    load_parameters_and_responses_from_runpath(str(run_path), prior_ensemble, [0])
    df = prior_ensemble.load_responses("gen_data", (0,))
    filter_cond = pl.col("report_step").eq(0), pl.col("values").is_not_nan()
    assert df.filter(filter_cond)["values"].to_list() == [1.0, 3.0]


def test_single_valued_gen_data_with_active_info_is_loaded(setup_case):
    config_text = dedent(
        """
    NUM_REALIZATIONS 1
    GEN_DATA RESPONSE RESULT_FILE:response_%d.out REPORT_STEPS:0
        """
    )
    prior_ensemble = setup_case(config_text)

    run_path = Path("simulations/realization-0/iter-0/")
    (run_path / "response_0.out").write_text("\n".join(["1"]), encoding="utf-8")
    (run_path / "response_0.out_active").write_text("\n".join(["1"]), encoding="utf-8")

    load_parameters_and_responses_from_runpath(str(run_path), prior_ensemble, [0])
    df = prior_ensemble.load_responses("RESPONSE", (0,))
    assert df["values"].to_list() == [1.0]


def test_that_all_deactivated_values_are_loaded(setup_case):
    config_text = dedent(
        """
    NUM_REALIZATIONS 1
    GEN_DATA RESPONSE RESULT_FILE:response_%d.out REPORT_STEPS:0
        """
    )
    prior_ensemble = setup_case(config_text)

    run_path = Path("simulations/realization-0/iter-0/")
    (run_path / "response_0.out").write_text("\n".join(["-1"]), encoding="utf-8")
    (run_path / "response_0.out_active").write_text("\n".join(["0"]), encoding="utf-8")

    load_parameters_and_responses_from_runpath(str(run_path), prior_ensemble, [0])
    response = prior_ensemble.load_responses("RESPONSE", (0,))
    assert np.isnan(response[0]["values"].to_list())
    assert len(response) == 1


@pytest.mark.usefixtures("use_tmpdir")
def test_loading_gen_data_without_restart(storage, run_args):
    config_text = dedent(
        """
    NUM_REALIZATIONS 1
    GEN_DATA RESPONSE RESULT_FILE:response.out
        """
    )
    Path("config.ert").write_text(config_text, encoding="utf-8")

    ert_config = ErtConfig.from_file("config.ert")
    prior_ensemble = storage.create_ensemble(
        storage.create_experiment(
            responses=ert_config.ensemble_config.response_configuration
        ),
        name="prior",
        ensemble_size=ert_config.runpath_config.num_realizations,
    )

    create_run_path(
        run_args=run_args(ert_config, prior_ensemble),
        ensemble=prior_ensemble,
        user_config_file=ert_config.user_config_file,
        env_vars=ert_config.env_vars,
        env_pr_fm_step=ert_config.env_pr_fm_step,
        forward_model_steps=ert_config.forward_model_steps,
        substitutions=ert_config.substitutions,
        parameters_file="parameters",
        runpaths=Runpaths.from_config(ert_config),
    )
    run_path = Path("simulations/realization-0/iter-0/")
    (run_path / "response.out").write_text("\n".join(["1", "2", "3"]), encoding="utf-8")
    (run_path / "response.out_active").write_text(
        "\n".join(["1", "0", "1"]), encoding="utf-8"
    )

    load_parameters_and_responses_from_runpath(str(run_path), prior_ensemble, [0])
    df = prior_ensemble.load_responses("RESPONSE", (0,))
    df_no_nans = df.filter(pl.col("values").is_not_nan())
    assert df_no_nans["values"].to_list() == [1.0, 3.0]


@pytest.mark.usefixtures("copy_snake_oil_case_storage")
@pytest.mark.integration_test
@pytest.mark.filterwarnings("ignore:Config contains a SUMMARY key")
def test_that_the_states_are_set_correctly():
    """
    When creating a new ensemble and loading results manually (load_from_forward_model)
    we expect only summary and gen_data to be copied and not parameters.
    """
    config = ErtConfig.from_file("snake_oil.ert")
    storage = open_storage(config.ens_path, mode="w")
    experiment = storage.get_experiment_by_name("ensemble-experiment")
    ensemble = experiment.get_ensemble_by_name("default_0")
    ensemble_size = ensemble.ensemble_size

    new_ensemble = storage.create_ensemble(
        experiment=ensemble.experiment, ensemble_size=ensemble_size
    )
    load_parameters_and_responses_from_runpath(
        config.runpath_config.runpath_format_string,
        new_ensemble,
        list(range(ensemble_size)),
    )

    assert all(
        RealizationStorageState.PARAMETERS_LOADED not in state
        for state in new_ensemble.get_ensemble_state()
    )

    assert new_ensemble.has_data()


@pytest.mark.parametrize("itr", [None, 0, 1, 2, 3])
@pytest.mark.usefixtures("use_tmpdir")
def test_loading_from_any_available_iter(storage, run_args, itr):
    ert_config = ErtConfig.from_file_contents(
        """\
        NUM_REALIZATIONS 1
        GEN_DATA RESPONSE RESULT_FILE:response.out
        """
    )
    prior_ensemble = storage.create_ensemble(
        storage.create_experiment(
            responses=ert_config.ensemble_config.response_configuration
        ),
        name="prior",
        ensemble_size=ert_config.runpath_config.num_realizations,
        iteration=itr if itr is not None else 0,
    )

    create_run_path(
        run_args=run_args(ert_config, prior_ensemble),
        ensemble=prior_ensemble,
        user_config_file=ert_config.user_config_file,
        env_vars=ert_config.env_vars,
        env_pr_fm_step=ert_config.env_pr_fm_step,
        forward_model_steps=ert_config.forward_model_steps,
        substitutions=ert_config.substitutions,
        parameters_file="parameters",
        runpaths=Runpaths.from_config(ert_config),
    )
    run_path = Path(f"simulations/realization-0/iter-{itr if itr is not None else 0}/")
    (run_path / "response.out").write_text("\n".join(["1", "2", "3"]), encoding="utf-8")
    (run_path / "response.out_active").write_text(
        "\n".join(["1", "0", "1"]), encoding="utf-8"
    )

    run_path_format = str(
        Path(
            f"simulations/realization-<IENS>/iter-{itr if itr is not None else 0}"
        ).resolve()
    )
    load_parameters_and_responses_from_runpath(run_path_format, prior_ensemble, [0])
    df = prior_ensemble.load_responses("RESPONSE", (0,))
    df_no_nans = df.filter(pl.col("values").is_not_nan())
    assert df_no_nans["values"].to_list() == [1.0, 3.0]
