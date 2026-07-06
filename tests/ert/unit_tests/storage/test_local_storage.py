import json
import logging
import os
import shutil
import stat
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import hypothesis.strategies as st
import numpy as np
import polars as pl
import pytest
from hypothesis import given

from ert.config import (
    BreakthroughConfig,
    SummaryConfig,
)
from ert.config._observations import (
    BreakthroughObservation,
)
from ert.dark_storage.common import ErtStoragePermissionError
from ert.storage import (
    ErtStorageException,
    LocalEnsemble,
    RealizationStorageState,
    open_storage,
)
from ert.storage.local_storage import _LOCAL_STORAGE_VERSION, LocalStorage
from ert.storage.mode import ModeError
from tests.ert.defaults_generator import (
    _create_summary_observation,
)
from tests.ert.unit_tests.storage._storage_test_helpers import (
    RaisingWriteNamedTemporaryFile,
)


def _ensembles(storage):
    return sorted(x.name for x in storage.ensembles)


def test_create_experiment(tmp_path):
    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment(name="test-experiment")

        experiment_path = storage.path / "experiments" / str(experiment.id)
        assert experiment_path.exists()

        with (experiment_path / "index.json").open(encoding="utf-8") as f:
            index = json.load(f)
            assert index["id"] == str(experiment.id)
            assert index["name"] == "test-experiment"


def test_that_loading_non_existing_experiment_throws(tmp_path):
    with (
        open_storage(tmp_path, mode="r") as storage,
        pytest.raises(
            KeyError, match="Experiment with name 'non-existing-experiment' not found"
        ),
    ):
        storage.get_experiment_by_name("non-existing-experiment")


def test_that_loading_non_existing_ensemble_throws(tmp_path):
    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment(name="test-experiment")
        with pytest.raises(
            KeyError, match="Ensemble with name 'non-existing-ensemble' not found"
        ):
            experiment.get_ensemble_by_name("non-existing-ensemble")


def test_open_empty_read(tmp_path):
    with open_storage(tmp_path / "empty", mode="r") as storage:
        assert _ensembles(storage) == []

    # Storage doesn't create an empty directory
    assert not (tmp_path / "empty").is_dir()


def test_open_empty_write(tmp_path):
    with open_storage(tmp_path / "empty", mode="w") as storage:
        assert _ensembles(storage) == []

    # Storage creates the directory
    assert (tmp_path / "empty").is_dir()


def test_open_with_no_permissions(tmp_path):
    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment()
        ensemble = storage.create_ensemble(experiment, name="foo", ensemble_size=1)
    path = Path(ensemble._path)
    mode = path.stat().st_mode
    Path(path).chmod(0o000)  # no permissions
    try:
        with (
            pytest.raises(
                ErtStoragePermissionError,
                match="Permission error when accessing storage at:",
            ),
            open_storage(tmp_path, mode="r") as storage,
        ):
            storage._load_experiments()
    finally:
        Path(path).chmod(mode)  # restore permissions


def test_reload(tmp_path):
    with open_storage(tmp_path, mode="w") as accessor:
        experiment_id = accessor.create_experiment()
        with open_storage(tmp_path, mode="r") as reader:
            assert _ensembles(accessor) == _ensembles(reader)

            accessor.create_ensemble(experiment_id, name="foo", ensemble_size=42)
            # Reader does not know about the newly created ensemble
            assert _ensembles(accessor) != _ensembles(reader)

            reader.reload()
            # Reader knows about it after the reload
            assert _ensembles(accessor) == _ensembles(reader)


def test_that_reader_storage_reads_most_recent_response_configs(tmp_path):
    reader = open_storage(tmp_path, mode="r")
    writer = open_storage(tmp_path, mode="w")

    exp = writer.create_experiment(
        experiment_config={
            "response_configuration": [
                SummaryConfig(
                    keys=["*", "FOPR"], input_files=["not_relevant"]
                ).model_dump(mode="json")
            ]
        },
        name="uniq",
    )
    ens: LocalEnsemble = exp.create_ensemble(ensemble_size=10, name="uniq_ens")

    reader.reload()
    read_exp = reader.get_experiment_by_name("uniq")
    assert read_exp.id == exp.id

    read_smry_config = read_exp.response_configuration["summary"]
    assert read_smry_config.keys == ["*", "FOPR"]
    assert not read_smry_config.has_finalized_keys

    smry_data = pl.DataFrame(
        {
            "response_key": ["FOPR", "FOPR", "WOPR", "WOPR", "FOPT", "FOPT"],
            "time": pl.Series(
                [datetime.now() + timedelta(days=i) for i in range(6)]  # noqa: DTZ005
            ).dt.cast_time_unit("ms"),
            "values": pl.Series([0.2, 0.2, 1.0, 1.1, 3.3, 3.3], dtype=pl.Float32),
        }
    )

    ens.save_response("summary", smry_data, 0)
    assert read_smry_config.keys == ["*", "FOPR"]
    assert not read_smry_config.has_finalized_keys

    reader.reload()
    read_exp = reader.get_experiment_by_name("uniq")
    read_smry_config = read_exp.response_configuration["summary"]
    assert read_smry_config.keys == ["FOPR", "FOPT", "WOPR"]
    assert read_smry_config.has_finalized_keys


def test_writing_to_read_only_storage_raises(tmp_path):
    with open_storage(tmp_path, mode="r") as storage, pytest.raises(ModeError):
        storage.create_experiment()


@pytest.mark.timeout(10)
def test_open_storage_write_with_empty_directory(tmp_path, caplog):
    with open_storage(tmp_path / "storage", mode="w") as storage:
        _ = storage.create_experiment()
        assert len(list(storage.experiments)) == 1

    with open_storage(tmp_path / "storage", mode="w") as storage:
        _ = storage.create_experiment()
        assert len(list(storage.experiments)) == 1

    storage.reload()
    assert len(list(storage.experiments)) == 0

    assert len(caplog.messages) == 0


def test_open_storage_read_with_empty_directory(tmp_path):
    with open_storage(tmp_path / "storage", mode="r") as storage:
        assert list(storage.ensembles) == []
        assert list(storage.experiments) == []


def test_open_storage_nested_dirs(tmp_path):
    with open_storage(tmp_path / "extra_level" / "storage", mode="w") as storage:
        assert storage.path.exists()


def test_open_storage_with_corrupted_storage(tmp_path):
    with open_storage(tmp_path / "storage", mode="w") as storage:
        storage.create_experiment().create_ensemble(name="prior", ensemble_size=1)
    os.remove(tmp_path / "storage" / "index.json")
    with pytest.raises(ErtStorageException, match="No index\\.json"):
        open_storage(tmp_path / "storage", mode="w")


def test_that_open_storage_in_read_mode_with_newer_version_throws_exception(tmp_path):
    with open_storage(tmp_path, mode="w") as storage:
        storage._index.version = _LOCAL_STORAGE_VERSION + 1
        storage._save_index()

    with pytest.raises(
        ErtStorageException,
        match=(
            f"Cannot open storage '{tmp_path}': "
            f"Storage version {_LOCAL_STORAGE_VERSION + 1} is newer than "
            f"the current version {_LOCAL_STORAGE_VERSION}, upgrade ert "
            "to continue, or run with a different ENSPATH"
        ),
    ):
        open_storage(tmp_path, mode="r")


def test_that_open_storage_in_read_mode_with_older_version_throws_exception(tmp_path):
    with open_storage(tmp_path, mode="w") as storage:
        storage._index.version = _LOCAL_STORAGE_VERSION - 1
        storage._save_index()

    with pytest.raises(
        ErtStorageException,
        match=(
            f"Cannot open storage '{tmp_path}' in read-only mode: "
            f"Storage version {_LOCAL_STORAGE_VERSION - 1} is too old"
        ),
    ):
        open_storage(tmp_path, mode="r")


def test_that_open_storage_in_write_mode_with_newer_version_throws_exception(tmp_path):
    with open_storage(tmp_path, mode="w") as storage:
        storage._index.version = _LOCAL_STORAGE_VERSION + 1
        storage._save_index()

    with pytest.raises(
        ErtStorageException,
        match=(
            f"Cannot open storage '{tmp_path}': "
            f"Storage version {_LOCAL_STORAGE_VERSION + 1} "
            f"is newer than the current version {_LOCAL_STORAGE_VERSION}, "
            "upgrade ert to continue, or run with a different ENSPATH"
        ),
    ):
        open_storage(tmp_path, mode="w")


def test_that_storage_migration_detected_and_completed(tmp_path):
    with open_storage(tmp_path, mode="w") as storage:
        storage._index.version = _LOCAL_STORAGE_VERSION - 1
        storage._save_index()

    assert LocalStorage.check_migration_needed(tmp_path)
    LocalStorage.perform_migration(tmp_path)
    assert not LocalStorage.check_migration_needed(tmp_path)

    with open_storage(tmp_path / "storage", mode="w") as storage:
        assert storage.version == _LOCAL_STORAGE_VERSION


def test_ensemble_no_parameters(storage):
    ensemble = storage.create_experiment(name="my-experiment").create_ensemble(
        ensemble_size=2,
        name="prior",
    )
    assert all(
        RealizationStorageState.RESPONSES_LOADED in s
        for s in ensemble.get_ensemble_state()
    )


@pytest.mark.filterwarnings("ignore:Config contains a SUMMARY key but no forward")
def test_get_unique_experiment_name(snake_oil_storage):
    with patch(
        "ert.storage.local_storage.LocalStorage.experiments", new_callable=PropertyMock
    ) as experiments:
        # Its not possible to do MagicMock(name="experiment_name") therefore
        # the workaround below
        names = [
            "experiment",
            "experiment_1",
            "experiment_8",
            "_d_e_",
            "___name__0___",
            "__name__1",
            "default",
        ]
        experiment_list = [MagicMock() for _ in range(len(names))]
        for k, v in zip(experiment_list, names, strict=False):
            k.name = v
        experiments.return_value = experiment_list

        assert snake_oil_storage.get_unique_experiment_name("_d_e_") == "_d_e__0"
        assert (
            snake_oil_storage.get_unique_experiment_name("experiment") == "experiment_9"
        )
        assert (
            snake_oil_storage.get_unique_experiment_name("___name__0___")
            == "___name__0____0"
        )
        assert snake_oil_storage.get_unique_experiment_name("name") == "name"
        assert snake_oil_storage.get_unique_experiment_name("__name__") == "__name__"
        assert snake_oil_storage.get_unique_experiment_name("") == "default_0"


def test_get_unique_experiment_name_ignores_non_numeric_suffixes(tmp_path):
    """
    Regression test for bug where non-numeric suffixes caused ValueError.

    When an experiment like 'es_mda_adaptive' existed, attempting to generate
    a unique name for 'es_mda' would incorrectly try to parse 'adaptive' as
    an integer, causing: ValueError: invalid literal for int() with base 10: 'adaptive'

    The fix ensures only numeric suffixes (e.g., 'es_mda_0', 'es_mda_1') are
    considered when generating new unique names.
    """
    with open_storage(tmp_path, mode="w") as storage:
        storage.create_experiment(name="es_mda")
        storage.create_experiment(name="es_mda_adaptive")

        # Should return es_mda_0, not crash on "adaptive"
        assert storage.get_unique_experiment_name("es_mda") == "es_mda_0"


@pytest.mark.usefixtures("use_tmpdir")
@given(st.binary())
def test_write_transaction(data):
    with open_storage(".", "w") as storage:
        filepath = Path("./file.txt")
        storage._write_transaction(filepath, data)

        assert stat.S_IMODE(filepath.stat().st_mode) == 0o660
        assert filepath.read_bytes() == data


def test_write_transaction_failure(tmp_path):
    with open_storage(tmp_path, "w") as storage:
        path = tmp_path / "file.txt"
        with (
            patch(
                "ert.storage.local_storage.NamedTemporaryFile",
                RaisingWriteNamedTemporaryFile,
            ) as f,
            pytest.raises(RuntimeError),
        ):
            storage._write_transaction(path, b"deadbeaf")

        assert f.entered

        assert not path.exists()


def test_write_transaction_overwrites(tmp_path):
    with open_storage(tmp_path, "w") as storage:
        path = tmp_path / "file.txt"
        path.write_text("abc")
        storage._write_transaction(path, b"deadbeaf")
        assert path.read_bytes() == b"deadbeaf"


@pytest.mark.parametrize(
    ("perturb_observations", "perturb_responses"),
    [
        pytest.param(
            False,
            True,
            id="Perturbed responses",
        ),
        pytest.param(
            True,
            False,
            id="Perturbed observations",
        ),
        pytest.param(
            True,
            False,
            id="Perturbed observations & responses",
        ),
    ],
)
def test_asof_joining_summary(tmp_path, perturb_observations, perturb_responses):
    with open_storage(tmp_path, mode="w") as storage:
        response_keys = ["FOPR", "FOPT_OP1", "FOPR:OP3", "FLAP", "F*"]
        obs_keys = [f"o_{k}" for k in response_keys]
        times = [datetime(2000, 1, 1, 1, 0)] * len(response_keys)  # noqa: DTZ001
        summary_observations = [
            {
                "type": "summary_observation",
                "name": obs_keys[i],
                "key": response_keys[i],
                "date": times[i].isoformat(),
                "value": 1.0,
                "error": 0.1,
            }
            for i in range(len(response_keys))
        ]

        experiment = storage.create_experiment(
            experiment_config={
                "response_configuration": [
                    SummaryConfig(keys=["*"], input_files=["not_relevant"]).model_dump(
                        mode="json"
                    )
                ],
                "observations": summary_observations,
            }
        )

        ensemble = storage.create_ensemble(
            experiment, ensemble_size=1, iteration=0, name="prior"
        )

        summary_df = pl.DataFrame(
            {
                "response_key": response_keys,
                "time": pl.Series(times, dtype=pl.Datetime("ms")),
                "values": pl.Series([0.0, 1.0, 2.0, 3.0, 4.0], dtype=pl.Float32),
            }
        )

        ensemble.save_response("summary", summary_df, 0)
        iens_active_index = np.array([0])

        obs_and_responses_exact = ensemble.get_observations_and_responses(
            obs_keys, iens_active_index
        )

        rng = np.random.default_rng(12345678)

        if perturb_responses:
            perturbed_summary = summary_df.with_columns(
                pl.when(pl.arange(0, summary_df.height) % 2 != 0)
                .then(pl.col("time") + pl.duration(milliseconds=rng.random() * 500))
                .otherwise(
                    pl.col("time") - pl.duration(milliseconds=rng.random() * 500)
                )
                .alias("time")
            )
            perturbed_summary = perturbed_summary.sort(by="time")
            ensemble.save_response("summary", perturbed_summary, 0)

        if perturb_observations:
            summary_obs_df = experiment.observations["summary"]
            perturbed_observations = summary_obs_df.with_columns(
                pl.when(pl.arange(0, summary_obs_df.height) % 2 != 0)
                .then(pl.col("time") + pl.duration(milliseconds=rng.random() * 500))
                .otherwise(
                    pl.col("time") - pl.duration(milliseconds=rng.random() * 500)
                )
                .alias("time")
            )
            perturbed_observations = perturbed_observations.sort(by="time")
            experiment.observations["summary"] = perturbed_observations

        obs_and_responses_perturbed = ensemble.get_observations_and_responses(
            obs_keys, iens_active_index
        )

        assert obs_and_responses_exact["0"].null_count() == 0
        assert obs_and_responses_perturbed["0"].null_count() == 0

        assert (
            obs_and_responses_exact.sort("response_key")
            .drop("index")
            .equals(obs_and_responses_perturbed.sort("response_key").drop("index"))
        )


def test_saving_everest_metadata_to_ensemble(tmp_path):
    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment()

        ensemble = storage.create_ensemble(
            experiment, ensemble_size=10, iteration=0, name="prior"
        )

        assert ensemble.everest_realization_info is None

        realization_info_dict = {
            ert_realization: {"model_realization": 0, "perturbation": -1}
            for ert_realization in range(10)
        }

        ensemble.save_everest_realization_info(realization_info_dict)
        assert ensemble.everest_realization_info == realization_info_dict


def test_that_saving_partial_everest_realization_info_raises_error(tmp_path):
    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment()

        ensemble = storage.create_ensemble(
            experiment, ensemble_size=10, iteration=0, name="prior"
        )

        with pytest.raises(
            ValueError,
            match=r"EVEREST realization info must describe all realizations.*[0, 2].*",
        ):
            ensemble.save_everest_realization_info({0: {}, 2: {}})


@pytest.mark.parametrize(
    "bad_realization_info",
    [
        {
            ert_realization: {"model_realization": None, "perturbation": -1}
            for ert_realization in range(10)
        },
        {
            ert_realization: {"model_realization": 0, "perturbation": None}
            for ert_realization in range(10)
        },
        {
            ert_realization: {"model_realization": 0, "perturbation": -2}
            for ert_realization in range(10)
        },
    ],
)
def test_that_saving_invalid_everest_realization_info_raises_error(
    tmp_path, bad_realization_info
):
    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment()

        ensemble = storage.create_ensemble(
            experiment, ensemble_size=10, iteration=0, name="prior"
        )

        with pytest.raises(ValueError, match="Bad everest realization info"):
            ensemble.save_everest_realization_info(bad_realization_info)


def test_set_failure_will_create_realization_directory(storage):
    # Setup
    dummy_ensemble = storage.create_experiment().create_ensemble(
        name="dummy", ensemble_size=1
    )
    assert dummy_ensemble._path.exists(), "Assumptions for test has changed"
    assert not (dummy_ensemble._path / "realization-0").exists()

    # Function test:
    dummy_ensemble.set_failure(0, RealizationStorageState.FAILURE_IN_CURRENT)

    # Then
    assert dummy_ensemble._path.glob(f"realization-0/{dummy_ensemble._error_log_name}")


def test_set_failure_will_not_recreate_ensemble_directory(storage):
    dummy_ensemble = storage.create_experiment().create_ensemble(
        name="dummy", ensemble_size=1
    )
    # Emulate a user deleting storage:
    shutil.rmtree(dummy_ensemble._path)

    # Then when a realization fails internalizing,
    dummy_ensemble.set_failure(0, RealizationStorageState.FAILURE_IN_CURRENT)

    # the ensemble path will not be created
    assert not dummy_ensemble._path.exists()


def test_that_permission_error_is_logged_in_load_ensembles(snake_oil_storage, caplog):
    ensemble = snake_oil_storage.get_experiment_by_name(
        "ensemble-experiment"
    ).get_ensemble_by_name("default_0")
    Path(ensemble._path).chmod(0o000)  # no permissions
    snake_oil_storage._ensembles.clear()
    try:
        with caplog.at_level(logging.ERROR), pytest.raises(PermissionError):
            snake_oil_storage._load_ensembles()
        assert (
            f"Permission error when loading ensemble from path: {ensemble._path}."
            in caplog.records[0].message
        )
        assert len(snake_oil_storage._ensembles) == 0
    finally:
        Path(ensemble._path).chmod(0o755)  # restore permissions


def test_that_permission_error_is_logged_in_load_index(snake_oil_storage, caplog):
    index_path = snake_oil_storage.path / "index.json"
    index_path.chmod(0o000)  # no permissions
    try:
        with caplog.at_level(logging.ERROR), pytest.raises(PermissionError):
            snake_oil_storage._load_index()
        assert (
            f"Permission error when loading index from path: {index_path}."
            in caplog.records[0].message
        )
    finally:
        index_path.chmod(0o744)  # restore permissions


def test_that_breakthrough_observations_and_responses_are_joined_in_endpoint(tmp_path):
    with open_storage(tmp_path, mode="w") as storage:
        response_key = "WWCT:OP1"
        # This observed time will be 1 day and 12 hours after derived breakthrough time
        time = datetime(2000, 3, 2, 13, 0)  # noqa: DTZ001

        breakthrough_config = BreakthroughConfig(
            keys=[f"BREAKTHROUGH:{response_key}"],
            summary_keys=[response_key],
            thresholds=[0.2],
            observed_dates=[time],
        )

        experiment = storage.create_experiment(
            experiment_config={
                "response_configuration": [
                    SummaryConfig(keys=["*"], input_files=["not_relevant"]).model_dump(
                        mode="json"
                    )
                ],
                "derived_response_configuration": [
                    breakthrough_config.model_dump(mode="json")
                ],
                "observations": [
                    BreakthroughObservation(
                        name="BRT_OP1",
                        key=response_key,
                        date=time,
                        error=10,
                        threshold=0.2,
                    ).model_dump(mode="json")
                ],
            }
        )

        ensemble = storage.create_ensemble(
            experiment, ensemble_size=1, iteration=0, name="prior"
        )

        response_dataframe = pl.DataFrame(
            {
                "realization": [0] * 10,
                "response_key": ["WWCT:OP1"] * 10,
                "time": [datetime(2000, month, 1, 1, 0) for month in range(1, 11)],  # noqa: DTZ001
                "values": [n / 10 for n in range(10)],
            }
        )
        ensemble.save_response("summary", response_dataframe, 0)

        breakthrough_derived_response = (
            ensemble.experiment.derived_response_configuration.get(
                "breakthrough"
            ).derive_from_storage(0, 0, ensemble)
        )

        ensemble.save_response("breakthrough", breakthrough_derived_response, 0)

        iens_active_index = np.array([0])

        obs_and_responses = ensemble.get_observations_and_responses(
            ["BRT_OP1"], iens_active_index
        )

        assert obs_and_responses["response_key"].to_list() == ["BREAKTHROUGH:WWCT:OP1"]
        assert obs_and_responses["observation_key"].to_list() == ["BRT_OP1"]
        assert obs_and_responses["index"].to_list() == ["0.2"]
        assert obs_and_responses["0"].to_list() == [-1.5]


def summary_response(*, key="FOPR", time="2000-01-01", values=100.0) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "response_key": [key],
            "time": pl.Series([datetime.fromisoformat(time)], dtype=pl.Datetime("ms")),
            "values": [values],
        },
    )


@pytest.mark.parametrize(
    ("summary_kwargs"),
    [
        pytest.param(
            {"key": "Fluffy"},
            id="on response key",
        ),
        pytest.param(
            {"time": "2024-01-01"},
            id="on match key",
        ),
    ],
)
def test_that_get_observations_and_responses_adds_qc_error_on_summary_mismatch(
    tmp_path, summary_kwargs
):
    with open_storage(tmp_path, mode="w") as storage:
        obs_date = datetime(2000, 1, 1)  # noqa: DTZ001

        summary_config = SummaryConfig(input_files=["not_relevant"], keys=["*"])

        summary_observation = _create_summary_observation(
            name="Summary_OBS",
            key="FOPR",
            date=obs_date.isoformat(),
        )

        experiment = storage.create_experiment(
            experiment_config={
                "response_configuration": [summary_config.model_dump(mode="json")],
                "observations": [
                    summary_observation.model_dump(mode="json"),
                ],
            }
        )

        ensemble = storage.create_ensemble(
            experiment, ensemble_size=2, iteration=0, name="prior"
        )

        ensemble.save_response("summary", summary_response(**summary_kwargs), 0)
        ensemble.save_response("summary", summary_response(**summary_kwargs), 1)

        iens_active_index = np.array([0, 1])
        active_observations = [
            "Summary_OBS",
        ]

        obs_and_responses = ensemble.get_observations_and_responses(
            active_observations, iens_active_index
        )

        assert obs_and_responses["observation_key"].to_list() == active_observations

        msg = (
            "no response matched observation data: "
            "response_key=FOPR, time=2000-01-01 00:00:00.000"
        )
        np.testing.assert_equal(obs_and_responses["0"].to_list(), [None])
        np.testing.assert_equal(obs_and_responses["1"].to_list(), [None])
        assert obs_and_responses["qc_error_0"].to_list() == [msg]
        assert obs_and_responses["qc_error_1"].to_list() == [msg]
