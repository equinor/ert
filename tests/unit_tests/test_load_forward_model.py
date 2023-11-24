import fileinput
import logging
import os
from datetime import datetime
from pathlib import Path
from textwrap import dedent

import numpy as np
import pytest
from resdata.summary import Summary

from ert.config import ErtConfig
from ert.enkf_main import create_run_path, ensemble_context
from ert.libres_facade import LibresFacade
from ert.realization_state import RealizationState
from ert.storage import open_storage


@pytest.fixture()
@pytest.mark.usefixtures("use_tmpdir")
def setup_case(storage):
    def func(config_text):
        Path("config.ert").write_text(config_text, encoding="utf-8")

        ert_config = ErtConfig.from_file("config.ert")
        prior_ensemble = storage.create_ensemble(
            storage.create_experiment(
                responses=ert_config.ensemble_config.response_configuration
            ),
            name="prior",
            ensemble_size=ert_config.model_config.num_realizations,
        )
        run_context = ensemble_context(
            prior_ensemble,
            [True],
            0,
            None,
            "",
            ert_config.model_config.runpath_format_string,
            "name",
        )
        create_run_path(run_context, ert_config.substitution_list, ert_config)
        return ert_config, prior_ensemble

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


@pytest.mark.skip(reason="Needs reimplementation")
@pytest.mark.usefixtures("copy_snake_oil_case_storage")
def test_load_inconsistent_time_map_summary(caplog):
    """
    Checking that we dont util_abort, we fail the forward model instead
    """
    cwd = os.getcwd()

    # Get rid of GEN_DATA as we are only interested in SUMMARY
    with fileinput.input("snake_oil.ert", inplace=True) as fin:
        for line in fin:
            if line.startswith("GEN_DATA"):
                continue
            print(line, end="")

    facade = LibresFacade.from_config_file("snake_oil.ert")
    realisation_number = 0
    storage = open_storage(facade.enspath, mode="w")
    ensemble = storage.get_ensemble_by_name("default_0")
    assert (
        ensemble.state_map[realisation_number] == RealizationState.HAS_DATA
    )  # Check prior state

    # Create a result that is incompatible with the refcase
    run_path = Path("storage") / "snake_oil" / "runpath" / "realization-0" / "iter-0"
    os.chdir(run_path)
    ecl_sum = run_simulator(1, datetime(2000, 1, 1))
    ecl_sum.fwrite()
    os.chdir(cwd)

    realizations = [False] * facade.get_ensemble_size()
    realizations[realisation_number] = True
    with caplog.at_level(logging.WARNING):
        loaded = facade.load_from_forward_model(ensemble, realizations, 0)
    assert (
        "Realization: 0, load warning: 200 inconsistencies in time map, first: "
        "Time mismatch for response time: 2010-01-10 00:00:00, last: Time mismatch "
        f"for response time: 2015-06-23 00:00:00 from: {run_path.absolute()}"
        f"/SNAKE_OIL_FIELD.UNSMRY"
    ) in caplog.messages
    assert loaded == 1


@pytest.mark.skip(reason="Needs reimplementation")
@pytest.mark.usefixtures("copy_snake_oil_case_storage")
def test_load_forward_model(snake_oil_default_storage):
    """
    Checking that we are able to load from forward model
    """
    facade = LibresFacade.from_config_file("snake_oil.ert")
    realisation_number = 0

    realizations = [False] * facade.get_ensemble_size()
    realizations[realisation_number] = True

    with open_storage(facade.enspath, mode="w") as storage:
        # 'load_from_forward_model' requires the ensemble to be writeable...
        default = storage.get_ensemble_by_name("default_0")

        loaded = facade.load_from_forward_model(default, realizations, 0)
        assert loaded == 1
        assert (
            default.state_map[realisation_number] == RealizationState.HAS_DATA
        )  # Check that status is as expected


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
            (0, "Could not find SUMMARY file"),
            id=(
                "Check that loading fails if we have configured"
                "SUMMARY but no summary is available in the run path"
            ),
        ),
    ],
)
def test_load_forward_model_summary(summary_configuration, storage, expected, caplog):
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

    run_context = ensemble_context(
        prior_ensemble,
        [True],
        0,
        None,
        "",
        ert_config.model_config.runpath_format_string,
        "name",
    )
    create_run_path(run_context, ert_config.substitution_list, ert_config)
    facade = LibresFacade(ert_config)
    with caplog.at_level(logging.ERROR):
        loaded = facade.load_from_forward_model(prior_ensemble, [True], 0)
    expected_loaded, expected_log_message = expected
    assert loaded == expected_loaded
    if expected_log_message:
        assert expected_log_message in "".join(caplog.messages)


@pytest.mark.usefixtures("use_tmpdir")
def test_load_forward_model_gen_data(setup_case):
    config_text = dedent(
        """
    NUM_REALIZATIONS 1
    GEN_DATA RESPONSE RESULT_FILE:response_%d.out REPORT_STEPS:0,1 INPUT_FORMAT:ASCII
        """
    )

    config, prior_ensemble = setup_case(config_text)
    run_path = Path("simulations/realization-0/iter-0/")
    with open(run_path / "response_0.out", "w", encoding="utf-8") as fout:
        fout.write("\n".join(["1", "2", "3"]))
    with open(run_path / "response_1.out", "w", encoding="utf-8") as fout:
        fout.write("\n".join(["4", "5", "5"]))
    with open(run_path / "response_0.out_active", "w", encoding="utf-8") as fout:
        fout.write("\n".join(["1", "0", "1"]))

    facade = LibresFacade(config)
    facade.load_from_forward_model(prior_ensemble, [True], 0)
    assert list(
        facade.load_gen_data(prior_ensemble, "RESPONSE", 0).dropna().values.flatten()
    ) == [1.0, 3.0]


@pytest.mark.usefixtures("use_tmpdir")
def test_single_valued_gen_data_with_active_info_is_loaded(setup_case):
    config_text = dedent(
        """
    NUM_REALIZATIONS 1
    GEN_DATA RESPONSE RESULT_FILE:response_%d.out REPORT_STEPS:0 INPUT_FORMAT:ASCII
        """
    )
    config, prior_ensemble = setup_case(config_text)

    run_path = Path("simulations/realization-0/iter-0/")
    with open(run_path / "response_0.out", "w", encoding="utf-8") as fout:
        fout.write("\n".join(["1"]))
    with open(run_path / "response_0.out_active", "w", encoding="utf-8") as fout:
        fout.write("\n".join(["1"]))

    facade = LibresFacade(config)
    facade.load_from_forward_model(prior_ensemble, [True], 0)
    assert list(
        facade.load_gen_data(prior_ensemble, "RESPONSE", 0).values.flatten()
    ) == [1.0]


@pytest.mark.usefixtures("use_tmpdir")
def test_that_all_decativated_values_are_loaded(setup_case):
    config_text = dedent(
        """
    NUM_REALIZATIONS 1
    GEN_DATA RESPONSE RESULT_FILE:response_%d.out REPORT_STEPS:0 INPUT_FORMAT:ASCII
        """
    )
    config, prior_ensemble = setup_case(config_text)

    run_path = Path("simulations/realization-0/iter-0/")
    with open(run_path / "response_0.out", "w", encoding="utf-8") as fout:
        fout.write("\n".join(["-1"]))
    with open(run_path / "response_0.out_active", "w", encoding="utf-8") as fout:
        fout.write("\n".join(["0"]))

    facade = LibresFacade(config)
    facade.load_from_forward_model(prior_ensemble, [True], 0)
    assert np.isnan(
        facade.load_gen_data(prior_ensemble, "RESPONSE", 0).values.flatten()[0]
    )
    assert (
        len(facade.load_gen_data(prior_ensemble, "RESPONSE", 0).values.flatten()) == 1
    )


@pytest.mark.usefixtures("use_tmpdir")
def test_loading_gen_data_without_restart(storage):
    config_text = dedent(
        """
    NUM_REALIZATIONS 1
    GEN_DATA RESPONSE RESULT_FILE:response.out INPUT_FORMAT:ASCII
        """
    )
    Path("config.ert").write_text(config_text, encoding="utf-8")

    ert_config = ErtConfig.from_file("config.ert")
    prior_ensemble = storage.create_ensemble(
        storage.create_experiment(
            responses=ert_config.ensemble_config.response_configuration
        ),
        name="prior",
        ensemble_size=ert_config.model_config.num_realizations,
    )

    run_context = ensemble_context(
        prior_ensemble,
        [True],
        0,
        None,
        "",
        ert_config.model_config.runpath_format_string,
        "name",
    )
    create_run_path(run_context, ert_config.substitution_list, ert_config)
    run_path = Path("simulations/realization-0/iter-0/")
    with open(run_path / "response.out", "w", encoding="utf-8") as fout:
        fout.write("\n".join(["1", "2", "3"]))
    with open(run_path / "response.out_active", "w", encoding="utf-8") as fout:
        fout.write("\n".join(["1", "0", "1"]))

    facade = LibresFacade.from_config_file("config.ert")
    facade.load_from_forward_model(prior_ensemble, [True], 0)
    assert list(
        facade.load_gen_data(prior_ensemble, "RESPONSE", 0).dropna().values.flatten()
    ) == [1.0, 3.0]
