import json
import logging
import os
import shutil
import stat
from datetime import date, datetime, timedelta
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
from ert.config.rft_config import RFTConfig
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
    _create_rft_observation,
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


def rft_response(
    *,
    well: tuple[str, ...] = ("WELL",),
    date: tuple[date, ...] = (datetime(2000, 1, 1).date(),),  # noqa: DTZ001
    prop: tuple[str, ...] = ("SWAT",),
    depth: tuple[float, ...] = (1006.6,),
    values: tuple[float, ...] = (100.0,),
    well_connection_cell: tuple[tuple[int, int, int], ...] = ((10, 10, 10),),
    cell_center: tuple[tuple[float, float, float], ...] = ((100.0, 105.0, 1000.0),),
    cell_zones: tuple[tuple[str, ...], ...] = (("zone1",),),
) -> pl.DataFrame:
    return (
        pl.DataFrame(
            {
                "well": well,
                "property": prop,
                "time": date,
                "depth": pl.Series(depth, dtype=pl.Float32),
                "values": pl.Series(values, dtype=pl.Float32),
                "well_connection_cell": pl.Series(
                    well_connection_cell, dtype=pl.Array(pl.Int64, 3)
                ),
                "cell_center": pl.Series(cell_center, dtype=pl.Array(pl.Float32, 3)),
                "cell_zones": pl.Series(cell_zones, dtype=pl.List(pl.String)),
            }
        )
        .with_columns(pl.col("time").dt.to_string("%Y-%m-%d").alias("date"))
        .with_columns(
            pl.concat_str(
                [pl.col("well"), pl.col("date"), pl.col("property")], separator=":"
            ).alias("response_key"),
        )
        .select(RFTConfig.response_schema().keys())
        .pipe(RFTConfig._assert_schema, RFTConfig.response_schema())
    )


def rft_observation(
    *,
    name="RFT_OBS1",
    well="WELL",
    date="2000-01-01",
    prop="PRESSURE",
    value=100.0,
    error=10.0,
    east=100.0,
    north=105.0,
    tvd=1000.0,
    zone=None,
):
    return _create_rft_observation(
        name=name,
        well=well,
        date=date,
        prop=prop,
        value=value,
        error=error,
        east=east,
        north=north,
        tvd=tvd,
        zone=zone,
    )


def rft_observation1(*, zone=None):
    return rft_observation(
        prop="SWAT",
        zone=zone,
    )


def rft_observation2(*, well="WELL", zone=None):
    return rft_observation(
        name="RFT_OBS2",
        well=well,
        prop="SWAT",
        east=300.0,
        north=405.0,
        tvd=2000.0,
        zone=zone,
    )


def location_metadata(
    *,
    east=(100.0, 300.0),
    north=(105.0, 405.0),
    tvd=(1000.0, 2000.0),
    actual_zones=(("zone1",), ("zone2",)),
    well_connection_cell=((10, 10, 10), (10, 10, 10)),
) -> pl.DataFrame:
    df = pl.DataFrame(
        {
            "east": east,
            "north": north,
            "tvd": tvd,
            "actual_zones": actual_zones,
            "well_connection_cell": well_connection_cell,
            "well_connection_cell_center": pl.Series(
                [(e, n, t) for e, n, t in zip(east, north, tvd, strict=True)],
                dtype=pl.Array(pl.Float32, 3),
            ),
        },
        schema=RFTConfig.location_metadata_schema(),
    )
    RFTConfig._assert_schema(df, RFTConfig.location_metadata_schema())
    return df


def test_that_get_observations_and_responses_applies_rft_metadata(tmp_path):
    with open_storage(tmp_path, mode="w") as storage:
        rft_config = RFTConfig(
            input_files=["BASE.RFT"],
            data_to_read={"WELL": {"2000-01-01": ["SWAT"]}},
        )

        obs1 = rft_observation1()
        obs2 = rft_observation2()

        experiment = storage.create_experiment(
            experiment_config={
                "response_configuration": [rft_config.model_dump(mode="json")],
                "observations": [
                    obs1.model_dump(mode="json"),
                    obs2.model_dump(mode="json"),
                ],
            }
        )

        ensemble = storage.create_ensemble(
            experiment, ensemble_size=2, iteration=0, name="prior"
        )

        ensemble.save_response("rft", rft_response(values=(200.0,)), 0)
        ensemble.save_observation_location_metadata(location_metadata(), 0)

        ensemble.save_response("rft", rft_response(values=(300.0,)), 1)
        ensemble.save_observation_location_metadata(location_metadata(), 1)

        iens_active_index = np.array([0, 1])
        active_observations = ["RFT_OBS1"]

        obs_and_responses = ensemble.get_observations_and_responses(
            active_observations, iens_active_index
        )

        assert obs_and_responses["response_key"].to_list() == ["WELL:2000-01-01:SWAT"]
        assert obs_and_responses["index"].to_list() == ["100.0, 105.0, 1000.0, None"]
        assert obs_and_responses["observation_key"].to_list() == active_observations
        assert obs_and_responses["0"].to_list() == [200.0]
        assert obs_and_responses["1"].to_list() == [300.0]


def test_that_get_observations_and_responses_disables_observation(tmp_path):
    with open_storage(tmp_path, mode="w") as storage:
        rft_config = RFTConfig(
            input_files=["BASE.RFT"],
            data_to_read={"WELL": {"2000-01-01": ["SWAT"]}},
        )

        obs1 = rft_observation1(zone="zone2")
        obs2 = rft_observation2(zone="zone3")

        experiment = storage.create_experiment(
            experiment_config={
                "response_configuration": [rft_config.model_dump(mode="json")],
                "observations": [
                    obs1.model_dump(mode="json"),
                    obs2.model_dump(mode="json"),
                ],
            }
        )

        ensemble = storage.create_ensemble(
            experiment, ensemble_size=2, iteration=0, name="prior"
        )

        ensemble.save_response(
            "rft", rft_response(values=(200.0,), cell_zones=(("zone1",),)), 0
        )
        ensemble.save_observation_location_metadata(
            location_metadata(actual_zones=(("zone1",), ("zone3",))), 0
        )

        ensemble.save_response(
            "rft", rft_response(values=(300.0,), cell_zones=(("zone2",),)), 1
        )
        ensemble.save_observation_location_metadata(
            location_metadata(actual_zones=(("zone2",), ("zone3",))), 1
        )

        iens_active_index = np.array([0, 1])
        active_observations = ["RFT_OBS1", "RFT_OBS2"]

        obs_and_responses = ensemble.get_observations_and_responses(
            active_observations, iens_active_index
        )

        assert obs_and_responses["observation_key"].to_list() == active_observations
        msg = "expected zone 'zone2' did not match any of the simulated zones: zone1"
        assert obs_and_responses["0"].to_list() == [None, 200.0]
        assert obs_and_responses["1"].to_list() == [300.0, 300.0]
        assert obs_and_responses["qc_error_0"].to_list() == [msg, None]
        assert obs_and_responses["qc_error_1"].to_list() == [None, None]


def test_that_get_observations_and_responses_combines_error_messages(tmp_path):
    with open_storage(tmp_path, mode="w") as storage:
        rft_config = RFTConfig(
            input_files=["BASE.RFT"],
            data_to_read={"WELL": {"2000-01-01": ["SWAT"]}},
        )

        obs1 = rft_observation1(zone="zone1")
        obs2 = rft_observation2(zone="zone2")

        experiment = storage.create_experiment(
            experiment_config={
                "response_configuration": [rft_config.model_dump(mode="json")],
                "observations": [
                    obs1.model_dump(mode="json"),
                    obs2.model_dump(mode="json"),
                ],
            }
        )

        ensemble = storage.create_ensemble(
            experiment, ensemble_size=2, iteration=0, name="prior"
        )

        ensemble.save_response(
            "rft", rft_response(values=(200.0,), cell_zones=(("zone1",),)), 0
        )
        ensemble.save_observation_location_metadata(
            location_metadata(
                actual_zones=(("zone0",), ("zone2",)),
                well_connection_cell=(None, (10, 10, 10)),
            ),
            0,
        )

        ensemble.save_response(
            "rft", rft_response(values=(300.0,), cell_zones=(("zone2",),)), 1
        )
        ensemble.save_observation_location_metadata(
            location_metadata(
                actual_zones=(("zone1",), ("zone2",)),
                well_connection_cell=(None, (10, 10, 10)),
            ),
            1,
        )

        iens_active_index = np.array([0, 1])
        active_observations = ["RFT_OBS1", "RFT_OBS2"]

        obs_and_responses = ensemble.get_observations_and_responses(
            active_observations, iens_active_index
        )

        assert obs_and_responses["observation_key"].to_list() == active_observations

        msg1 = "expected zone 'zone1' did not match any of the simulated zones: zone0"
        msg2 = "did not find grid coordinate for location 100.0, 105.0, 1000.0"
        msg3 = (
            "no response matched observation data: "
            "response_key=WELL:2000-01-01:SWAT, well_connection_cell=None"
        )
        msg_real0 = f"{msg1};\n{msg2};\n{msg3}"
        msg_real1 = f"{msg2};\n{msg3}"
        assert obs_and_responses["0"].to_list() == [None, 200.0]
        assert obs_and_responses["1"].to_list() == [None, 300.0]
        assert obs_and_responses["qc_error_0"].to_list() == [msg_real0, None]
        assert obs_and_responses["qc_error_1"].to_list() == [msg_real1, None]


@pytest.mark.parametrize(
    ("obs2_kwargs", "response_kwargs", "meta_kwargs"),
    [
        pytest.param(
            {"well": "OTHER_WELL"},
            {"well": ("OTHER_WELL",)},
            {},
            id="on response key",
        ),
        pytest.param(
            {},
            {"well_connection_cell": ([11, 11, 11],)},
            {"well_connection_cell": ([10, 10, 10], [11, 11, 11])},
            id="on match key",
        ),
    ],
)
def test_that_get_observations_and_responses_adds_qc_error_on_rft_mismatch(
    tmp_path, obs2_kwargs, response_kwargs, meta_kwargs
):
    with open_storage(tmp_path, mode="w") as storage:
        rft_config = RFTConfig(
            input_files=["BASE.RFT"],
            data_to_read={"WELL": {"2000-01-01": ["SWAT"]}},
        )

        obs1 = rft_observation1()
        obs2 = rft_observation2(**obs2_kwargs)

        experiment = storage.create_experiment(
            experiment_config={
                "response_configuration": [rft_config.model_dump(mode="json")],
                "observations": [
                    obs1.model_dump(mode="json"),
                    obs2.model_dump(mode="json"),
                ],
            }
        )

        ensemble = storage.create_ensemble(
            experiment, ensemble_size=2, iteration=0, name="prior"
        )

        ensemble.save_response("rft", rft_response(**response_kwargs), 0)
        ensemble.save_response("rft", rft_response(**response_kwargs), 1)

        ensemble.save_observation_location_metadata(location_metadata(**meta_kwargs), 0)
        ensemble.save_observation_location_metadata(location_metadata(**meta_kwargs), 1)

        iens_active_index = np.array([0, 1])
        active_observations = ["RFT_OBS1", "RFT_OBS2"]

        obs_and_responses = ensemble.get_observations_and_responses(
            active_observations, iens_active_index
        )

        assert obs_and_responses["observation_key"].to_list() == active_observations

        msg = (
            "no response matched observation data: "
            "response_key=WELL:2000-01-01:SWAT, well_connection_cell=[10, 10, 10]"
        )
        np.testing.assert_equal(obs_and_responses["0"].to_list(), [None, 100.0])
        np.testing.assert_equal(obs_and_responses["1"].to_list(), [None, 100.0])
        assert obs_and_responses["qc_error_0"].to_list() == [msg, None]
        assert obs_and_responses["qc_error_1"].to_list() == [msg, None]


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


@pytest.mark.parametrize(
    (
        "approximate_missing_values",
        "missing_required_columns",
        "expected_response_values",
    ),
    [
        pytest.param(True, False, [110.0, 310.0], id="interpolate enabled"),
        pytest.param(False, False, [None, None], id="interpolate disabled"),
        pytest.param(True, True, [None, None], id="missing required columns"),
    ],
)
def test_that_get_observations_and_responses_interpolates_rft_values(
    tmp_path,
    approximate_missing_values,
    missing_required_columns,
    expected_response_values,
):
    with open_storage(tmp_path, mode="w") as storage:
        rft_config = RFTConfig(
            input_files=["BASE.RFT"],
            data_to_read={"WELL": {"2000-01-01": ["PRESSURE"]}},
            approximate_missing_values=approximate_missing_values,
        )

        obs1 = rft_observation(name="RFT_OBS1", value=100.0, tvd=1000.0, zone="zone1")
        obs2 = rft_observation(name="RFT_OBS2", value=200.0, tvd=2000.0, zone="zone1")

        experiment = storage.create_experiment(
            experiment_config={
                "response_configuration": [rft_config.model_dump(mode="json")],
                "observations": [
                    obs1.model_dump(mode="json"),
                    obs2.model_dump(mode="json"),
                ],
            }
        )

        ensemble = storage.create_ensemble(
            experiment, ensemble_size=1, iteration=0, name="prior"
        )

        date = datetime(2000, 1, 1).date()  # noqa: DTZ001

        # If the response is missing the required columns for interpolation,
        # we should skip interpolation and return None, even if interpolation is
        # enabled. The test drops the columns to simulate a legacy response without
        # these columns.
        drop_columns = ["cell_center", "cell_zones"] if missing_required_columns else []
        ensemble.save_response(
            "rft",
            rft_response(
                well=("WELL", "WELL"),
                date=(date, date),
                prop=("PRESSURE", "PRESSURE"),
                depth=(500.0, 1500.0),
                values=(50.0, 150.0),
                well_connection_cell=((10, 10, 10), (10, 10, 12)),
                cell_center=((100.0, 105.0, 700.0), (100.0, 105.0, 1200.0)),
                cell_zones=(("zone1",), ("zone1",)),
            ).drop(drop_columns),
            0,
        )
        ensemble.save_observation_location_metadata(
            location_metadata(
                east=(100.0, 100.0),
                north=(105.0, 105.0),
                tvd=(1000.0, 2000.0),
                actual_zones=(("zone1",), ("zone1",)),
                well_connection_cell=((10, 10, 11), (10, 10, 13)),
            ),
            0,
        )

        iens_active_index = np.array([0])
        active_observations = ["RFT_OBS1", "RFT_OBS2"]

        obs_and_responses = ensemble.get_observations_and_responses(
            active_observations, iens_active_index
        )

        assert obs_and_responses["response_key"].to_list() == [
            "WELL:2000-01-01:PRESSURE",
            "WELL:2000-01-01:PRESSURE",
        ]
        np.testing.assert_array_equal(
            obs_and_responses["0"].to_list(), expected_response_values
        )
