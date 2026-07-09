import json

import pytest

from ert.storage.migration.to36 import migrate


def verify_migration(tmp_path, original_experiment_data, expected_experiment_data):
    root = tmp_path / "project"
    root.mkdir()

    exp_path = root / "experiments" / "exp1"
    exp_path.mkdir(parents=True)

    index_data = {
        "id": "exp-id",
        "name": "exp1",
        "ensembles": [],
        "experiment": {
            **original_experiment_data,
        },
    }
    (exp_path / "index.json").write_text(json.dumps(index_data), encoding="utf-8")

    migrate(root)

    updated = json.loads((exp_path / "index.json").read_text(encoding="utf-8"))
    assert updated["experiment"] == expected_experiment_data


def test_that_migration_to_36_combines_response_config_and_derived_response_config(
    tmp_path,
):
    summary_config = {
        "type": "summary",
        "input_files": ["SPE1"],
        "keys": ["FGOR", "FOPR"],
        "has_finalized_keys": "true",
    }
    rft_config = {
        "type": "rft",
        "input_files": ["SPE1"],
        "keys": ["PROD:2015-02-01:PRESSURE", "PROD:2017-07-31:PRESSURE"],
        "has_finalized_keys": "true",
        "name": "rft",
        "data_to_read": {
            "PROD": {
                "2015-02-01": ["PRESSURE"],
                "2016-01-31": ["PRESSURE"],
                "2017-07-31": ["PRESSURE"],
            }
        },
        "zonemap": "zonemap.txt",
        "approximate_missing_values": "false",
    }

    breakthrough_config = {
        "type": "breakthrough",
        "keys": ["BREAKTHROUGH:FGOR"],
        "has_finalized_keys": "true",
        "summary_keys": ["FGOR"],
        "thresholds": [1.5],
        "observed_dates": ["2021-03-31T00:00:00"],
    }

    original_experiment_data = {
        "response_configuration": [summary_config, rft_config],
        "derived_response_configuration": [breakthrough_config],
    }

    expected_experiment_data = {
        "response_configuration": [summary_config, rft_config, breakthrough_config],
    }

    verify_migration(tmp_path, original_experiment_data, expected_experiment_data)


@pytest.mark.parametrize(
    "experiment_data",
    [
        pytest.param({}, id="no response configuration"),
        pytest.param({"response_configuration": []}, id="empty response configuration"),
        pytest.param(
            {"response_configuration": [{"key": "value"}]},
            id="response configuration has value",
        ),
    ],
)
def test_that_migration_to_36_does_nothing_on_missing_derived_response_configuration(
    tmp_path, experiment_data
):
    original_experiment_data = experiment_data
    expected_experiment_data = original_experiment_data

    assert "derived_response_configuration" not in original_experiment_data
    assert "derived_response_configuration" not in expected_experiment_data

    verify_migration(tmp_path, original_experiment_data, expected_experiment_data)


@pytest.mark.parametrize(
    "experiment_data",
    [
        pytest.param({}, id="no response configuration"),
        pytest.param({"response_configuration": []}, id="empty response configuration"),
        pytest.param(
            {"response_configuration": [{"key": "value"}]},
            id="response configuration has value",
        ),
    ],
)
def test_that_migration_to_36_removes_empty_derived_response_configuration_entry(
    tmp_path, experiment_data
):
    original_experiment_data = experiment_data.copy()
    original_experiment_data.update({"derived_response_configuration": []})
    expected_experiment_data = experiment_data

    assert "derived_response_configuration" in original_experiment_data
    assert "derived_response_configuration" not in expected_experiment_data

    verify_migration(tmp_path, original_experiment_data, expected_experiment_data)


@pytest.mark.parametrize(
    "experiment_data",
    [
        pytest.param(
            {"derived_response_configuration": [{"key": "value"}]},
            id="derived response config has value, no response configuration",
        ),
        pytest.param(
            {
                "response_configuration": [],
                "derived_response_configuration": [{"key": "value"}],
            },
            id="derived response config has value, but response configuration is empty",
        ),
    ],
)
def test_that_migration_to_36_populates_empty_response_configuration(
    tmp_path, experiment_data
):
    original_experiment_data = experiment_data
    expected_experiment_data = {
        "response_configuration": [{"key": "value"}],
    }

    verify_migration(tmp_path, original_experiment_data, expected_experiment_data)


def test_that_migration_to_36_does_not_fail_on_unexpectedly_structured_dirs(tmp_path):

    root = tmp_path / "project"
    root.mkdir()
    migrate(root)

    experiments_dir = root / "experiments"
    experiments_dir.mkdir()

    not_an_experiment = experiments_dir / "not_a_directory.json"
    not_an_experiment.write_text("{}", encoding="utf-8")

    migrate(root)

    experiment_without_index_json = experiments_dir / "experiment"
    experiment_without_index_json.mkdir()

    migrate(root)
