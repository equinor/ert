import os

import numpy as np

from ert.config import GenDataConfig, SummaryConfig
from ert.config.standardized_response_config import (
    StandardResponseConfig,
)
from ert.storage import open_storage
from tests.performance_tests.performance_utils import (
    write_summary_data,
    write_summary_spec,
)


def test_that_summary_to_standard_response_config_works():
    summary_config = SummaryConfig(
        name="summary", input_file="summaryOUT", keys=["A", "B", "C", "D", "E"]
    )
    standardized_configs_list = StandardResponseConfig.standardize_configs(
        [summary_config]
    )
    assert len(standardized_configs_list) == 1
    standardized = standardized_configs_list[0]

    assert standardized.keys == ["A", "B", "C", "D", "E"]
    assert standardized.input_files == ["summaryOUT"]
    assert standardized.cardinality == "one_per_realization"
    assert standardized.response_type == "summary"


def test_that_multiple_gendatas_to_standard_response_config_works():
    alt1 = GenDataConfig(
        name="ALT1", input_file="f1", kwargs={"report_steps": [2, 1, 3]}
    )
    alt2 = GenDataConfig(
        name="ALT2", input_file="f2", kwargs={"report_steps": [2, 3, 1]}
    )
    alt3 = GenDataConfig(name="ALT3", input_file="f3", kwargs={"report_steps": [3]})
    alt4 = GenDataConfig(name="ALT4", input_file="f4", kwargs={"report_steps": [3]})
    alt5 = GenDataConfig(name="ALT5", input_file="f5", kwargs={"report_steps": [4]})

    standardized_configs_list = StandardResponseConfig.standardize_configs(
        [alt1, alt2, alt3, alt4, alt5]
    )

    assert len(standardized_configs_list) == 1
    standardized_config = standardized_configs_list[0]
    assert standardized_config.keys == ["ALT1", "ALT2", "ALT3", "ALT4", "ALT5"]
    assert standardized_config.input_files == ["f1", "f2", "f3", "f4", "f5"]
    assert standardized_config.cardinality == "one_per_key"
    assert standardized_config.response_type == "gen_data"


def test_that_single_gendata_to_standard_response_config_works():
    standardized_configs_list = StandardResponseConfig.standardize_configs(
        [
            GenDataConfig(
                name="ALT1", input_file="f1", kwargs={"report_steps": [2, 1, 3]}
            )
        ]
    )

    assert len(standardized_configs_list) == 1
    standardized_config = standardized_configs_list[0]
    assert standardized_config.keys == ["ALT1"]
    assert standardized_config.input_files == ["f1"]
    assert standardized_config.cardinality == "one_per_key"
    assert standardized_config.response_type == "gen_data"


def test_read_and_combine_gendata_from_runpath(tmp_path):
    alt1 = GenDataConfig(
        name="ALT1", input_file="f1@%d", kwargs={"report_steps": [2, 1, 3]}
    )
    alt2 = GenDataConfig(
        name="ALT2", input_file="f2@%d", kwargs={"report_steps": [2, 3, 1]}
    )
    alt3 = GenDataConfig(name="ALT3", input_file="f3@%d", kwargs={"report_steps": [3]})
    alt4 = GenDataConfig(name="ALT4", input_file="f4@%d", kwargs={"report_steps": [3]})
    gendata_list = [alt1, alt2, alt3, alt4]

    standardized_configs_list = StandardResponseConfig.standardize_configs(gendata_list)

    assert len(standardized_configs_list) == 1
    standardized_config = standardized_configs_list[0]
    assert standardized_config.keys == ["ALT1", "ALT2", "ALT3", "ALT4"]
    assert standardized_config.input_files == [
        "f1@%d",
        "f2@%d",
        "f3@%d",
        "f4@%d",
    ]
    assert standardized_config.cardinality == "one_per_key"
    assert standardized_config.response_type == "gen_data"

    run_path = tmp_path / "iter-0" / "realization-0"
    os.mkdir(tmp_path / "iter-0")
    os.mkdir(run_path)

    lines_per_config = {
        alt1.name: 1,
        alt2.name: 20,
        alt3.name: 30,
        alt4.name: 40,
    }

    # Create some gen data files in the runpath
    for config in [alt1, alt2, alt3, alt4]:
        num_lines = lines_per_config[config.name]
        for report_step in config.report_steps or [""]:
            file_path = run_path / (f"{config.input_file}" % report_step)

            some_data = np.linspace(
                100 * report_step,
                99 + 100 * report_step,
                num_lines,
            )

            with open(file_path, "w+", encoding="utf-8") as f:
                np.savetxt(f, some_data, fmt="%f")

    combined_ds = standardized_config.read_from_file(run_path=run_path, iens=0)

    # Approximation for correctness:
    # Assert that the non-nan line count for each report step
    # matches up with the expected lines per report step
    all_report_steps = set.union(*[set(config.report_steps) for config in gendata_list])
    for report_step in all_report_steps:
        expected_n_lines = sum(
            lines_per_config[config.name]
            for config in gendata_list
            if report_step in config.report_steps
        )

        assert (
            len(combined_ds.sel(report_step=report_step).to_dataframe().dropna())
            == expected_n_lines
        )


def test_read_and_combine_gendata_from_runpath_without_report_step(tmp_path):
    alt5 = GenDataConfig(name="ALT5", input_file="f5@%d")
    alt6 = GenDataConfig(name="ALT6", input_file="f6@")
    gendata_list = [alt5, alt6]

    standardized_configs_list = StandardResponseConfig.standardize_configs(gendata_list)

    assert len(standardized_configs_list) == 1
    standardized_config = standardized_configs_list[0]
    assert standardized_config.keys == ["ALT5", "ALT6"]
    assert standardized_config.input_files == [
        "f5@%d",
        "f6@",
    ]
    assert standardized_config.cardinality == "one_per_key"
    assert standardized_config.response_type == "gen_data"

    run_path = tmp_path / "iter-0" / "realization-0"
    os.mkdir(tmp_path / "iter-0")
    os.mkdir(run_path)

    lines_per_config = {
        "ALT5": 101,
        "ALT6": 101,
    }

    # Create some gen data files in the runpath
    for config in [alt5, alt6]:
        num_lines = lines_per_config[config.name]
        file_path = run_path / config.input_file

        some_data = np.linspace(
            100,
            199,
            num_lines,
        )

        with open(file_path, "w+", encoding="utf-8") as f:
            np.savetxt(f, some_data, fmt="%f")

    combined_ds = standardized_config.read_from_file(run_path=run_path, iens=0)

    expected_n_lines = sum(lines_per_config[config.name] for config in gendata_list)

    assert len(combined_ds.to_dataframe().dropna()) == expected_n_lines


def test_reading_summary_through_standardized_spec(tmp_path):
    summary_keys = ["PSUMA", "PSUMB", "PSUMC", "PSUMD", "PSUME"]
    num_timesteps = 100
    summary_config = SummaryConfig(
        name="summary", input_file="summaryOUT", keys=summary_keys
    )

    standardized_configs_list = StandardResponseConfig.standardize_configs(
        [summary_config]
    )
    assert len(standardized_configs_list) == 1
    standardized = standardized_configs_list[0]

    run_path = tmp_path / "iter-0" / "realization-0"
    os.mkdir(tmp_path / "iter-0")
    os.mkdir(run_path)

    write_summary_spec(run_path / "summaryOUT.SMSPEC", summary_keys)
    write_summary_data(
        run_path / "summaryOUT.UNSMRY",
        num_timesteps,
        summary_keys,
        3,
    )

    ds = standardized.read_from_file(run_path, 0)
    assert ds["name"].data.tolist() == summary_keys

    for _, _ds in ds.groupby("name"):
        assert _ds["values"].size == num_timesteps


def test_standard_response_json_writing(tmp_path):
    summary_keys = ["PSUMA", "PSUMB", "PSUMC", "PSUMD", "PSUME"]
    summary_config = SummaryConfig(
        name="summary", input_file="summaryOUT", keys=summary_keys
    )

    alt1 = GenDataConfig(
        name="ALT1", input_file="f1@%d", kwargs={"report_steps": [2, 1, 3]}
    )
    alt2 = GenDataConfig(
        name="ALT2", input_file="f2@%d", kwargs={"report_steps": [2, 3, 1]}
    )
    alt3 = GenDataConfig(name="ALT3", input_file="f3@%d", kwargs={"report_steps": [3]})
    alt4 = GenDataConfig(name="ALT4", input_file="f4@%d", kwargs={"report_steps": [3]})
    alt5 = GenDataConfig(name="ALT5", input_file="f5@%d")
    gendata_list = [alt1, alt2, alt3, alt4, alt5]

    with open_storage(tmp_path / "storage", mode="w") as storage:
        exp = storage.create_experiment(responses=[*gendata_list, summary_config])
        response_config = exp.response_configuration
        assert (
            response_config["summary"]
            == StandardResponseConfig.standardize_configs([summary_config])[0]
        )

        assert (
            response_config["gen_data"]
            == StandardResponseConfig.standardize_configs(gendata_list)[0]
        )
