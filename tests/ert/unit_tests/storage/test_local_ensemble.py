import asyncio
import io
import json
import logging
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import hypothesis.strategies as st
import numpy as np
import polars as pl
import pytest
import scipy.sparse as sp
from hypothesis import given
from pydantic import ValidationError

from ert.analysis.event import AnalysisCompleteEvent, AnalysisMatrixEvent, DataSection
from ert.config import GenKwConfig, RFTConfig, SummaryConfig
from ert.config.response_config import InvalidResponseFile
from ert.exceptions import StorageError
from ert.storage import LocalExperiment, open_storage
from ert.storage.blob_data import (
    BlobStorageData,
    BlobType,
    MatrixStorageData,
    ObservationReportData,
)
from ert.storage.local_ensemble import (
    _write_responses_to_storage,
)
from ert.storage.mode import ModeError
from tests.ert.defaults_generator import _create_rft_observation


def test_that_load_scalar_keys_loads_all_parameters(tmp_path):
    """Test that load_scalar_keys loads all scalar parameters when keys=None."""
    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment(
            experiment_config={
                "parameter_configuration": [
                    GenKwConfig(
                        name="param1",
                        group="group1",
                        distribution={"name": "uniform", "min": 0, "max": 1},
                    ).model_dump(mode="json"),
                    GenKwConfig(
                        name="param2",
                        group="group1",
                        distribution={"name": "uniform", "min": 0, "max": 1},
                    ).model_dump(mode="json"),
                    GenKwConfig(
                        name="param3",
                        group="group2",
                        distribution={"name": "normal", "mean": 0, "std": 1},
                    ).model_dump(mode="json"),
                ]
            }
        )
        ensemble = storage.create_ensemble(experiment.id, ensemble_size=5, name="test")

        # Save parameters
        ensemble.save_parameters(
            pl.DataFrame(
                {
                    "realization": [0, 1, 2],
                    "param1": [1.0, 2.0, 3.0],
                    "param2": [4.0, 5.0, 6.0],
                    "param3": [7.0, 8.0, 9.0],
                }
            )
        )

        # Load all parameters
        df = ensemble.load_scalar_keys()
        assert df.shape == (3, 4)
        assert "realization" in df.columns
        assert "param1" in df.columns
        assert "param2" in df.columns
        assert "param3" in df.columns
        assert df["param1"].to_list() == [1.0, 2.0, 3.0]


def test_that_load_scalar_keys_loads_specific_parameters(tmp_path):
    """Test that load_scalar_keys loads only specified parameters."""
    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment(
            experiment_config={
                "parameter_configuration": [
                    GenKwConfig(
                        name="param1",
                        group="group1",
                        distribution={"name": "uniform", "min": 0, "max": 1},
                    ).model_dump(mode="json"),
                    GenKwConfig(
                        name="param2",
                        group="group1",
                        distribution={"name": "uniform", "min": 0, "max": 1},
                    ).model_dump(mode="json"),
                    GenKwConfig(
                        name="param3",
                        group="group2",
                        distribution={"name": "normal", "mean": 0, "std": 1},
                    ).model_dump(mode="json"),
                ]
            }
        )
        ensemble = storage.create_ensemble(experiment.id, ensemble_size=5, name="test")

        ensemble.save_parameters(
            pl.DataFrame(
                {
                    "realization": [0, 1, 2],
                    "param1": [1.0, 2.0, 3.0],
                    "param2": [4.0, 5.0, 6.0],
                    "param3": [7.0, 8.0, 9.0],
                }
            )
        )

        # Load only param1 and param3
        df = ensemble.load_scalar_keys(keys=["param1", "param3"])
        assert df.shape == (3, 3)
        assert "realization" in df.columns
        assert "param1" in df.columns
        assert "param3" in df.columns
        assert "param2" not in df.columns


def test_that_load_scalar_keys_filters_by_realizations(tmp_path):
    """Test that load_scalar_keys filters by specified realizations."""
    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment(
            experiment_config={
                "parameter_configuration": [
                    GenKwConfig(
                        name="param1",
                        group="group1",
                        distribution={"name": "uniform", "min": 0, "max": 1},
                    ).model_dump(mode="json"),
                ]
            }
        )
        ensemble = storage.create_ensemble(experiment.id, ensemble_size=5, name="test")

        ensemble.save_parameters(
            pl.DataFrame(
                {
                    "realization": [0, 1, 2, 3, 4],
                    "param1": [1.0, 2.0, 3.0, 4.0, 5.0],
                }
            )
        )

        # Load only realizations 1 and 3
        df = ensemble.load_scalar_keys(keys=["param1"], realizations=np.array([1, 3]))
        assert df.shape == (2, 2)
        assert df["realization"].to_list() == [1, 3]
        assert df["param1"].to_list() == [2.0, 4.0]


def test_that_load_scalar_keys_raises_key_error_for_missing_parameters(tmp_path):
    """Test that load_scalar_keys raises KeyError for non-existent parameters."""
    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment(
            experiment_config={
                "parameter_configuration": [
                    GenKwConfig(
                        name="param1",
                        group="group1",
                        distribution={"name": "uniform", "min": 0, "max": 1},
                    ).model_dump(mode="json"),
                ]
            }
        )
        ensemble = storage.create_ensemble(experiment.id, ensemble_size=5, name="test")

        with pytest.raises(KeyError, match="No SCALAR dataset in storage"):
            ensemble.load_scalar_keys(keys=["param1"])


def test_that_load_scalar_keys_raises_key_error_for_unregistered_parameters(tmp_path):
    """Test that load_scalar_keys raises KeyError for parameters not in experiment."""
    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment(
            experiment_config={
                "parameter_configuration": [
                    GenKwConfig(
                        name="param1",
                        group="group1",
                        distribution={"name": "uniform", "min": 0, "max": 1},
                    ).model_dump(mode="json"),
                ]
            }
        )
        ensemble = storage.create_ensemble(experiment.id, ensemble_size=5, name="test")

        ensemble.save_parameters(
            pl.DataFrame(
                {
                    "realization": [0, 1, 2],
                    "param1": [1.0, 2.0, 3.0],
                }
            )
        )

        with pytest.raises(
            KeyError,
            match="Parameters not registered to the experiment: \\{'param2'\\}",
        ):
            ensemble.load_scalar_keys(keys=["param1", "param2"])


def test_that_load_scalar_keys_raises_index_error_for_missing_realizations(tmp_path):
    """Test that load_scalar_keys raises IndexError when no matching realizations."""
    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment(
            experiment_config={
                "parameter_configuration": [
                    GenKwConfig(
                        name="param1",
                        group="group1",
                        distribution={"name": "uniform", "min": 0, "max": 1},
                    ).model_dump(mode="json"),
                ]
            }
        )
        ensemble = storage.create_ensemble(experiment.id, ensemble_size=5, name="test")

        ensemble.save_parameters(
            pl.DataFrame(
                {
                    "realization": [0, 1, 2],
                    "param1": [1.0, 2.0, 3.0],
                }
            )
        )

        with pytest.raises(
            IndexError,
            match="No matching realizations \\[5 6\\] found for \\['param1'\\]",
        ):
            ensemble.load_scalar_keys(keys=["param1"], realizations=np.array([5, 6]))


def _create_rft_response_df(
    *,
    well: str = "WELL1",
    date: str = "2020-01-01",
    prop: str = "PRESSURE",
    depth: float = 8000.0,
    value: float = 148.0,
    i: int = 1,
    j: int = 2,
    k: int = 3,
    cell_center: tuple[float, float, float] = (100.0, 200.0, 25.0),
    cell_zones: tuple[str, ...] = (),
) -> pl.DataFrame:
    time = datetime.strptime(date, "%Y-%m-%d").date()  # noqa: DTZ007
    df = pl.DataFrame(
        {
            "response_key": [f"{well}:{date}:{prop}"],
            "well": [well],
            "date": [date],
            "property": [prop],
            "time": [time],
            "depth": pl.Series([depth], dtype=pl.Float32),
            "values": pl.Series([value], dtype=pl.Float32),
            "well_connection_cell": pl.Series([(i, j, k)], dtype=pl.Array(pl.Int64, 3)),
            "cell_center": pl.Series([cell_center], dtype=pl.Array(pl.Float32, 3)),
            "cell_zones": pl.Series([cell_zones], dtype=pl.List(pl.String)),
        }
    )
    return RFTConfig._assert_schema(df, RFTConfig.response_schema())


def _create_rft_location_metadata_df(
    *,
    east: float = 100.0,
    north: float = 200.0,
    tvd: float = 25.0,
    zone: str | None = None,
    i: int = 1,
    j: int = 2,
    k: int = 3,
) -> pl.DataFrame:
    zones = [zone] if zone is not None else []
    df = pl.DataFrame(
        {
            "east": pl.Series([east], dtype=pl.Float32),
            "north": pl.Series([north], dtype=pl.Float32),
            "tvd": pl.Series([tvd], dtype=pl.Float32),
            "actual_zones": pl.Series([zones], dtype=pl.List(pl.String)),
            "well_connection_cell": pl.Series([(i, j, k)], dtype=pl.Array(pl.Int64, 3)),
            "well_connection_cell_center": pl.Series(
                [(east, north, tvd)], dtype=pl.Array(pl.Float32, 3)
            ),
        }
    )
    return RFTConfig._assert_schema(df, RFTConfig.location_metadata_schema())


@contextmanager
def _create_rft_ensemble(
    ensemble_size,
    observations,
    *,
    with_summary=False,
    zonemap=None,
    approximate_missing_values=False,
):
    if zonemap is not None:
        zonemap_file = Path("zonemap.txt")
        zonemap_file.write_text(zonemap, encoding="utf-8")
    else:
        zonemap_file = None
    rft_config = RFTConfig(
        input_files=["DUMMY"],
        zonemap=zonemap_file,
        approximate_missing_values=approximate_missing_values,
    )
    result_config = [rft_config.model_dump(mode="json")]

    if with_summary:
        summary_config = SummaryConfig(keys=["FOPR"], input_files=["DUMMY"])
        result_config.append(summary_config.model_dump(mode="json"))

    with open_storage("storage", mode="w") as storage:
        experiment = storage.create_experiment(
            experiment_config={
                "response_configuration": result_config,
                "observations": [o.model_dump(mode="json") for o in observations],
            }
        )
        yield storage.create_ensemble(
            experiment.id, ensemble_size=ensemble_size, name="test"
        )


@pytest.mark.usefixtures("use_tmpdir")
def test_that_get_rft_observations_and_responses_returns_joined_data():
    with _create_rft_ensemble(
        1, [_create_rft_observation(zone="Z1")], zonemap="1 Z1"
    ) as ensemble:
        realization = 0
        ensemble.save_response(
            "rft",
            pl.concat(
                [
                    _create_rft_response_df(prop="PRESSURE", value=148.0),
                    _create_rft_response_df(prop="SGAS", value=0.1),
                    _create_rft_response_df(prop="SWAT", value=0.2),
                ]
            ),
            realization,
        )
        ensemble.save_observation_location_metadata(
            _create_rft_location_metadata_df(zone="Z1"),
            realization,
        )

        result = ensemble.get_rft_observations_and_responses()

        assert {k: v.to_list() for k, v in result.to_dict().items()} == {
            "order": [0],
            "utm_x": [100.0],
            "utm_y": [200.0],
            "measured_depth": [50.0],
            "true_vertical_depth": [25.0],
            "zone": ["Z1"],
            "pressure": [148.0],
            "swat": [pytest.approx(0.2)],
            "sgas": [pytest.approx(0.1)],
            "soil": [pytest.approx(0.7)],  # soil is computed as 1 - sgas - swat
            "valid_zone": [True],
            "is_active": [True],
            "i": [1],
            "j": [2],
            "k": [3],
            "well": ["WELL1"],
            "time": ["2020-01-01"],
            "realization": [0],
            "report_step": [0],
            "observed": [150.0],
            "error": [5.0],
        }


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.parametrize(
    ("approximate_missing_values", "expected_approximated_values"),
    [
        (True, 165.0),
        (False, None),
    ],
)
def test_that_get_rft_observations_and_responses_includes_approximated_values_when_enabled(  # noqa: E501
    approximate_missing_values, expected_approximated_values
):
    observation = _create_rft_observation(zone="zone1")
    with _create_rft_ensemble(
        1, [observation], approximate_missing_values=approximate_missing_values
    ) as ensemble:
        realization = 0
        # Save two responses that does not match the observation, but can be used to
        # approximate a response at the observation location.
        ensemble.save_response(
            "rft",
            pl.concat(
                [
                    _create_rft_response_df(
                        i=0,
                        j=0,
                        k=1,
                        cell_center=(100.0, 200.0, 20.0),
                        cell_zones=("zone1",),
                        value=160.0,
                    ),
                    _create_rft_response_df(
                        i=0,
                        j=0,
                        k=3,
                        cell_center=(100.0, 200.0, 30.0),
                        cell_zones=("zone1",),
                        value=170.0,
                    ),
                ]
            ),
            realization,
        )
        ensemble.save_observation_location_metadata(
            _create_rft_location_metadata_df(i=0, j=0, k=2, zone="zone1"),
            realization,
        )
        result = ensemble.get_rft_observations_and_responses()

        # The approximated response value is in the result dataframe:
        assert result["pressure"].to_list() == [
            pytest.approx(expected_approximated_values)
        ]


@pytest.mark.usefixtures("use_tmpdir")
def test_that_get_rft_observations_is_active_based_on_matching_pressure_response():
    """Test that is_active is True when there is a matching PRESSURE response,
    and False if not
    """
    observations = [
        _create_rft_observation(),
        _create_rft_observation(
            name="obs2",
            tvd=30.0,
            md=60.0,
            value=160.0,
        ),
    ]

    with _create_rft_ensemble(1, observations) as ensemble:
        realization = 0
        ensemble.save_response(
            "rft", pl.concat([_create_rft_response_df()]), realization
        )
        ensemble.save_observation_location_metadata(
            _create_rft_location_metadata_df(), realization
        )

        result = ensemble.get_rft_observations_and_responses().sort(
            "true_vertical_depth"
        )

        assert result["is_active"].to_list() == [
            True,
            False,
        ]  # tvd=25 has pressure response, tvd=30 does not


@pytest.mark.usefixtures("use_tmpdir")
def test_that_get_rft_observations_and_responses_sets_valid_zone_with_null_equality():
    observations = [
        _create_rft_observation(zone="Z1"),
        _create_rft_observation(
            name="obs2",
            tvd=30.0,
            md=60.0,
            value=160.0,
        ),
        _create_rft_observation(
            name="obs3",
            tvd=35.0,
            md=70.0,
            zone="Z2",
            value=170.0,
        ),
    ]

    realization = 0
    with _create_rft_ensemble(1, observations, zonemap="1 Z1\n2 Z2\n") as ensemble:
        ensemble.save_response(
            "rft",
            _create_rft_response_df(),
            realization,
        )
        ensemble.save_observation_location_metadata(
            pl.concat(
                [
                    _create_rft_location_metadata_df(zone="Z1", tvd=25.0),
                    _create_rft_location_metadata_df(zone=None, tvd=30.0),
                    _create_rft_location_metadata_df(zone="Z1", tvd=35.0),
                ]
            ),
            realization,
        )

        result = ensemble.get_rft_observations_and_responses().sort(
            "true_vertical_depth"
        )

        assert result["valid_zone"][0] is True  # Z1 == Z1
        assert result["valid_zone"][1] is True  # None == None
        assert result["valid_zone"][2] is False  # Z2 != Z1


@pytest.mark.usefixtures("use_tmpdir")
def test_that_get_rft_observations_and_responses_order_is_row_index_within_well():
    observations = [
        _create_rft_observation(name="obs1", tvd=25.0, md=50.0),
        _create_rft_observation(name="obs2", tvd=30.0, md=60.0),
        _create_rft_observation(name="obs3", tvd=35.0, md=70.0),
        _create_rft_observation(well="WELL2", name="obs4", tvd=40.0, md=80.0),
        _create_rft_observation(well="WELL2", name="obs5", tvd=45.0, md=90.0),
    ]

    with _create_rft_ensemble(1, observations) as ensemble:
        realization = 0
        ensemble.save_response(
            "rft", pl.concat([_create_rft_response_df()]), realization
        )
        ensemble.save_observation_location_metadata(
            _create_rft_location_metadata_df(),
            realization,
        )

        result = ensemble.get_rft_observations_and_responses().sort(
            ["well", "true_vertical_depth"]
        )

        assert result["well"].to_list() == ["WELL1", "WELL1", "WELL1", "WELL2", "WELL2"]
        assert result["order"].to_list() == [0, 1, 2, 0, 1]


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.slow
@given(
    response_values=st.lists(
        st.floats(allow_infinity=False, allow_nan=False, width=32), min_size=1
    ),
    prop=st.sampled_from(["PRESSURE", "SWAT", "SGAS"]),
)
def test_that_get_rft_observations_and_responses_handles_multiple_realizations(
    response_values: list[float], prop: str
):
    with _create_rft_ensemble(
        len(response_values), [_create_rft_observation()]
    ) as ensemble:
        for i, r in enumerate(response_values):
            ensemble.save_response(
                "rft", _create_rft_response_df(value=r, prop=prop), i
            )
            ensemble.save_observation_location_metadata(
                _create_rft_location_metadata_df(),
                i,
            )

        result = ensemble.get_rft_observations_and_responses().sort("realization")

        assert result.shape[0] == len(response_values)
        assert result["realization"].to_list() == list(range(len(response_values)))

        for i, r in enumerate(response_values):
            assert result[prop.lower()][i] == pytest.approx(r)


@pytest.mark.usefixtures("use_tmpdir")
def test_that_get_rft_observations_and_responses_raises_error_for_no_observations():
    with (
        _create_rft_ensemble(1, []) as ensemble,
        pytest.raises(StorageError, match="No RFT observations found"),
    ):
        ensemble.get_rft_observations_and_responses()


@pytest.mark.usefixtures("use_tmpdir")
def test_that_get_rft_observations_and_responses_raises_error_when_response_not_saved():
    with (
        _create_rft_ensemble(1, [_create_rft_observation()]) as ensemble,
        pytest.raises(KeyError, match="No response for key rft"),
    ):
        ensemble.get_rft_observations_and_responses()


@pytest.mark.usefixtures("use_tmpdir")
def test_that_get_rft_observations_and_responses_raises_error_when_location_metadata_not_saved():  # noqa: E501
    with _create_rft_ensemble(1, [_create_rft_observation()]) as ensemble:
        ensemble.save_response("rft", _create_rft_response_df(), 0)
        with pytest.raises(FileNotFoundError, match="observation_location_metadata"):
            ensemble.get_rft_observations_and_responses()


@pytest.mark.usefixtures("use_tmpdir")
def test_that_get_rft_observations_and_responses_adds_missing_saturation_columns():
    realization = 0
    with _create_rft_ensemble(1, [_create_rft_observation()]) as ensemble:
        # Save a response that does not contain saturations
        ensemble.save_response("rft", _create_rft_response_df(), realization)
        ensemble.save_observation_location_metadata(
            _create_rft_location_metadata_df(),
            realization,
        )

        result = ensemble.get_rft_observations_and_responses()

        for saturation in ["sgas", "swat", "soil"]:
            assert saturation in result.columns
            assert result[saturation].to_list() == [None]


@pytest.mark.usefixtures("use_tmpdir")
def test_that_get_rft_observations_and_responses_maps_report_step_from_summary_times(
    tmp_path,
):
    observations = [
        _create_rft_observation(date="2020-01-15"),
        _create_rft_observation(
            date="2020-02-15",
            name="obs2",
            tvd=30.0,
            md=60.0,
            value=160.0,
        ),
    ]

    rft_responses = pl.concat(
        [_create_rft_response_df(value=148.0), _create_rft_response_df(value=158.0)]
    )

    summary_responses = pl.DataFrame(
        {
            "response_key": ["FOPR"] * 3,
            "time": pl.Series(
                [datetime(2020, 1, 1), datetime(2020, 1, 15), datetime(2020, 2, 15)]  # noqa: DTZ001
            ).dt.cast_time_unit("ms"),
            "values": pl.Series([100.0, 200.0, 300.0], dtype=pl.Float32),
        }
    )

    with _create_rft_ensemble(1, observations, with_summary=True) as ensemble:
        realization = 0
        ensemble.save_response("rft", rft_responses, realization)
        ensemble.save_response("summary", summary_responses, realization)
        ensemble.save_observation_location_metadata(
            _create_rft_location_metadata_df(),
            realization,
        )

        result = ensemble.get_rft_observations_and_responses().sort("time")

        assert "report_step" in result.columns
        assert result["report_step"].to_list() == [1, 2]


@pytest.mark.usefixtures("use_tmpdir")
def test_that_rft_artifacts_are_saved_atomically(caplog):
    caplog.set_level(logging.WARNING)

    def read_from_file_mock(run_path, iens, iter_) -> pl.DataFrame:
        if iens == 1:
            raise InvalidResponseFile("Mock error for realization 1")
        if iens == 3:
            raise Exception("Mock error for realization 3")
        return _create_rft_response_df()

    def location_metadata_mock(run_path, iens, iter_, observations) -> pl.DataFrame:
        if iens == 2:
            raise InvalidResponseFile("Mock error for realization 2")
        if iens == 4:
            raise Exception("Mock error for realization 4")
        return _create_rft_location_metadata_df()

    async def run_test():
        with (
            patch.object(RFTConfig, "read_from_file", side_effect=read_from_file_mock),
            patch.object(
                RFTConfig,
                "obtain_location_metadata",
                side_effect=location_metadata_mock,
            ),
        ):
            num_realizations = 5
            observations = [
                _create_rft_observation(),
            ]

            with _create_rft_ensemble(num_realizations, observations) as ensemble:
                results = []
                for realization in range(num_realizations):
                    result = await _write_responses_to_storage(
                        "", realization, ensemble
                    )
                    results.append(result)

                assert results[0].successful
                for realization in range(1, num_realizations):
                    assert not results[realization].successful

                logs = [
                    (r.levelno, " ".join(r.getMessage().split()[:5]))
                    for r in caplog.records
                ]
                assert logs == [
                    (logging.WARNING, "Failed to read response from"),
                    (logging.WARNING, "Failed to write observation metadata"),
                    (logging.ERROR, "Unexpected exception while reading from"),
                    (logging.ERROR, "Unexpected exception while writing RFT"),
                ]

                ensemble.load_responses("rft", (0,))
                for realization in range(1, num_realizations):
                    with pytest.raises(KeyError):
                        ensemble.load_responses("rft", (realization,))

                ensemble.load_observation_location_metadata(0)
                for realization in range(1, num_realizations):
                    with pytest.raises(FileNotFoundError):
                        ensemble.load_observation_location_metadata(realization)

                assert ensemble.get_realization_list_with_responses() == [0]

    asyncio.run(run_test())


@pytest.mark.usefixtures("use_tmpdir")
def test_that_location_metadata_file_is_not_created_when_no_observations():
    def read_from_file_mock(run_path, iens, iter_):
        return _create_rft_response_df()

    async def run_test():
        with (
            patch.object(RFTConfig, "read_from_file", side_effect=read_from_file_mock),
        ):
            num_realizations = 1
            observations = []

            with _create_rft_ensemble(num_realizations, observations) as ensemble:
                await _write_responses_to_storage("", 0, ensemble)

                ensemble.load_responses("rft", (0,))
                with pytest.raises(FileNotFoundError):
                    ensemble.load_observation_location_metadata(0)
                assert ensemble.get_realization_list_with_responses() == [0]

    asyncio.run(run_test())


def test_that_save_blob_raises_in_read_mode(tmp_path):
    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment()
        storage.create_ensemble(experiment, ensemble_size=1, iteration=0, name="prior")

    with open_storage(tmp_path, mode="r") as storage:
        ensemble = next(iter(storage.ensembles))
        event = AnalysisCompleteEvent(
            data=DataSection(
                header=["x"],
                data=[(1,)],
            ),
            update_algorithm="ensemble_smoother",
        )
        with pytest.raises(ModeError):
            ensemble.save_blob(event)


def test_that_observation_report_blob_writes_parquet_metadata_and_can_be_loaded(
    tmp_path,
):
    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment()
        ensemble = storage.create_ensemble(
            experiment, ensemble_size=1, iteration=0, name="prior"
        )

        event = AnalysisCompleteEvent(
            data=DataSection(
                header=["observation_key", "status", "value"],
                data=[
                    ("OBS_1", "Active", 1.5),
                    ("OBS_2", "Deactivated, outlier", 2.0),
                ],
            ),
            update_algorithm="ensemble_smoother",
        )
        ensemble.save_blob(event)

        blob_dir = ensemble._path / "blobs"

        assert blob_dir.is_dir(), "Expected blob directory to be created"

        blob_files = list(blob_dir.glob("*.blob"))
        json_files = list(blob_dir.glob("*.blob.json"))
        assert len(blob_files) == 1
        assert len(json_files) == 1

        loaded_df = pl.read_parquet(blob_files[0])
        assert loaded_df.columns == ["observation_key", "status", "value"]
        assert len(loaded_df) == 2
        assert loaded_df["observation_key"][0] == "OBS_1"

        metadata = json.loads(json_files[0].read_text(encoding="utf-8"))
        assert metadata["blob_info"]["blob_type"] == "observation_report"
        assert metadata["blob_info"]["update_algorithm"] == "ensemble_smoother"
        assert metadata["name"] == "observation_report"
        assert metadata["file_size"] > 0

        loaded_blobs = ensemble.load_blobs(BlobType.OBSERVATION_REPORT)
        assert len(loaded_blobs) == 1
        assert isinstance(loaded_blobs[0], BlobStorageData)
        assert isinstance(loaded_blobs[0].blob_info, ObservationReportData)
        assert loaded_blobs[0].blob_info.update_algorithm == "ensemble_smoother"
        assert loaded_blobs[0].name == "observation_report"
        assert loaded_blobs[0].file_type == "application/parquet"


@pytest.mark.parametrize(
    ("blob_event", "expected_exception"),
    [
        pytest.param(
            AnalysisMatrixEvent.model_construct(
                event_type="AnalysisMatrixEvent",
                name="bad_shape",
                sparse=False,
                shape=(2,),
                data_type="float64",
                update_algorithm="enif",
                matrix_bytes=b"matrix-bytes",
            ),
            ValidationError,
            id="matrix-shape-has-one-dimension",
        ),
        pytest.param(
            AnalysisMatrixEvent.model_construct(
                event_type="AnalysisMatrixEvent",
                name="bad_bytes",
                sparse=False,
                shape=(1, 1),
                data_type="float64",
                update_algorithm="enif",
                matrix_bytes="matrix-bytes",
            ),
            TypeError,
            id="matrix-bytes-are-text",
        ),
        pytest.param(
            AnalysisCompleteEvent.model_construct(
                event_type="AnalysisCompleteEvent",
                data={"header": ["x"], "data": [(1,)]},
                update_algorithm="ensemble_smoother",
            ),
            AttributeError,
            id="observation-data-is-a-dict",
        ),
    ],
)
def test_that_save_blob_rejects_malformed_blob_events(
    tmp_path,
    blob_event,
    expected_exception,
):
    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment()
        ensemble = storage.create_ensemble(
            experiment, ensemble_size=1, iteration=0, name="prior"
        )

        with pytest.raises(expected_exception):
            ensemble.save_blob(blob_event)

        blob_dir = ensemble._path / "blobs"
        assert not list(blob_dir.glob("*.blob"))
        assert not list(blob_dir.glob("*.json"))


@pytest.mark.parametrize(
    "metadata",
    [
        pytest.param(b"{", id="invalid-json"),
        pytest.param(
            json.dumps(
                {
                    "uri": "metadata.blob",
                    "file_size": 1,
                    "file_type": "application/parquet",
                    "name": "observation_report",
                }
            ).encode("utf-8"),
            id="missing-blob-info",
        ),
        pytest.param(
            json.dumps(
                {
                    "uri": "metadata.blob",
                    "file_size": 1,
                    "file_type": "application/parquet",
                    "name": "observation_report",
                    "blob_info": {
                        "blob_type": "unknown",
                        "update_algorithm": "ensemble_smoother",
                    },
                }
            ).encode("utf-8"),
            id="unknown-blob-type",
        ),
    ],
)
def test_that_load_blobs_rejects_malformed_blob_metadata(
    tmp_path,
    metadata,
):
    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment()
        ensemble = storage.create_ensemble(
            experiment, ensemble_size=1, iteration=0, name="prior"
        )
        blob_dir = ensemble._path / "blobs"
        blob_dir.mkdir()
        (blob_dir / "metadata.blob.json").write_bytes(metadata)

        with pytest.raises(ValidationError):
            ensemble.load_blobs()


@pytest.mark.parametrize(
    "uri_template",
    [
        pytest.param("missing.blob", id="missing-file"),
        pytest.param("../index.json", id="parent-directory"),
        pytest.param("{ensemble_path}/index.json", id="absolute-path"),
    ],
)
def test_that_load_blob_raises_file_not_found_for_invalid_uri(
    tmp_path,
    uri_template,
):
    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment()
        ensemble = storage.create_ensemble(
            experiment, ensemble_size=1, iteration=0, name="prior"
        )
        uri = uri_template.format(ensemble_path=ensemble._path)

        with pytest.raises(FileNotFoundError):
            ensemble.load_blob(uri)


def test_that_sparse_and_dense_matrix_blobs_can_be_saved_and_loaded(tmp_path):

    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment()
        ensemble = storage.create_ensemble(
            experiment, ensemble_size=1, iteration=0, name="prior"
        )

        # Create a sparse matrix and serialize it
        sparse_matrix = sp.csc_array(np.array([[1.0, 0.0], [0.0, 2.0]]))
        sparse_buf = io.BytesIO()
        sp.save_npz(sparse_buf, sparse_matrix)
        sparse_bytes = sparse_buf.getvalue()

        sparse_event = AnalysisMatrixEvent(
            name="H",
            sparse=True,
            shape=(2, 2),
            data_type="float64",
            update_algorithm="enif",
            matrix_bytes=sparse_bytes,
        )
        ensemble.save_blob(sparse_event)

        # Create a dense matrix and serialize it
        dense_matrix = np.array([[3.0, 4.0], [5.0, 6.0]])
        dense_buf = io.BytesIO()
        np.save(dense_buf, dense_matrix)
        dense_bytes = dense_buf.getvalue()

        dense_event = AnalysisMatrixEvent(
            name="Prec_u",
            sparse=False,
            shape=(2, 2),
            data_type="float64",
            update_algorithm="enif",
            matrix_bytes=dense_bytes,
        )
        ensemble.save_blob(dense_event)

        # load_blobs returns all matrix metadata
        all_blobs = ensemble.load_blobs(BlobType.MATRIX)
        assert len(all_blobs) == 2
        assert all(isinstance(m, BlobStorageData) for m in all_blobs)
        assert all(isinstance(m.blob_info, MatrixStorageData) for m in all_blobs)

        by_name = {m.name: m for m in all_blobs}
        assert "H" in by_name
        assert "Prec_u" in by_name

        h_meta = by_name["H"]
        assert isinstance(h_meta.blob_info, MatrixStorageData)
        assert h_meta.blob_info.sparse is True
        assert h_meta.blob_info.shape == (2, 2)
        assert h_meta.file_type == "application/x-npz"

        prec_meta = by_name["Prec_u"]
        assert isinstance(prec_meta.blob_info, MatrixStorageData)
        assert prec_meta.blob_info.sparse is False
        assert prec_meta.blob_info.shape == (2, 2)
        assert prec_meta.file_type == "application/x-npy"

        h_bytes = ensemble.load_blob(h_meta.uri)
        loaded_sparse = sp.load_npz(io.BytesIO(h_bytes))
        np.testing.assert_array_equal(loaded_sparse.toarray(), sparse_matrix.toarray())

        prec_bytes = ensemble.load_blob(prec_meta.uri)
        loaded_dense = np.load(io.BytesIO(prec_bytes))
        np.testing.assert_array_equal(loaded_dense, dense_matrix)

        assert h_meta.blob_info.parameter_group_sizes == {}
        assert prec_meta.blob_info.parameter_group_sizes == {}


def test_that_parameter_group_sizes_is_stored_in_matrix_blob_metadata(tmp_path):
    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment()
        ensemble = storage.create_ensemble(
            experiment, ensemble_size=1, iteration=0, name="prior"
        )

        dense_matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
        dense_buf = io.BytesIO()
        np.save(dense_buf, dense_matrix)

        event = AnalysisMatrixEvent(
            name="K",
            sparse=False,
            shape=(2, 2),
            data_type="float64",
            update_algorithm="enif",
            parameter_group_sizes={"PORO": 8, "PERM": 3},
            matrix_bytes=dense_buf.getvalue(),
        )
        ensemble.save_blob(event)

        blobs = ensemble.load_blobs(BlobType.MATRIX)
        assert len(blobs) == 1
        assert isinstance(blobs[0].blob_info, MatrixStorageData)
        assert blobs[0].blob_info.parameter_group_sizes == {"PORO": 8, "PERM": 3}


async def test_that_writing_and_reading_empty_response_in_storage_results_in_empty_df_with_schema_columns(  # noqa: E501
    tmp_path, monkeypatch
):
    """This test writes an empty set of responses to storage and asserts that the
    parquet file contains the correct columns.
    Then the test also checks that loading said file through ensemble results in an
    empty dataframe.
    """
    response_column_scheme = ["realization", "response_key", "time", "values"]
    empty_response = pl.DataFrame({"response_key": [], "time": [], "values": []})
    monkeypatch.setattr(SummaryConfig, "read_from_file", lambda *args: empty_response)
    monkeypatch.setattr(
        LocalExperiment, "response_configuration", {"summary": SummaryConfig()}
    )

    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment()
        ensemble = storage.create_ensemble(experiment.id, ensemble_size=1, name="test")
        await _write_responses_to_storage(str(tmp_path), 0, ensemble)

        summary_response_path = ensemble._realization_dir(0) / "summary.parquet"
        assert Path(summary_response_path).is_file()
        responses = pl.read_parquet(summary_response_path)
        assert responses.is_empty()
        assert responses.columns == response_column_scheme

        # Mock response config to contain a response key, else the parquet file
        # will never be read as the code exits earlier.
        monkeypatch.setattr(
            LocalExperiment,
            "response_configuration",
            {"summary": SummaryConfig(keys=["FOPR"])},
        )
        monkeypatch.setattr(
            LocalExperiment, "response_key_to_response_type", {"FOPR": "summary"}
        )

        responses = ensemble.load_responses("FOPR", (0,))
        assert responses.is_empty()
        assert responses.columns == response_column_scheme
