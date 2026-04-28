import asyncio
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import hypothesis.strategies as st
import numpy as np
import polars as pl
import pytest
from hypothesis import given

from ert.config import GenKwConfig, RFTConfig, SummaryConfig
from ert.config._observations import RFTObservation
from ert.config.response_config import InvalidResponseFile
from ert.exceptions import StorageError
from ert.storage import open_storage
from ert.storage.local_ensemble import _write_responses_to_storage
from ert.storage.mode import ModeError


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


def _create_rft_observation(
    well: str = "WELL1",
    date: str = "2020-01-01",
    prop: str = "PRESSURE",
    obs_name: str = "obs1",
    east: float = 100.0,
    north: float = 200.0,
    tvd: float = 25.0,
    md: float | None = 50.0,
    zone: str | None = None,
    value: float = 150.0,
    error: float = 5.0,
) -> RFTObservation:
    return RFTObservation(
        name=obs_name,
        well=well,
        date=date,
        property=prop,
        value=value,
        error=error,
        north=north,
        east=east,
        tvd=tvd,
        md=md,
        zone=zone,
    )


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
        }
    )
    return RFTConfig._assert_schema(df, RFTConfig.location_metadata_schema())


@contextmanager
def _create_rft_ensemble(ensemble_size, observations, with_summary=False, zonemap=None):
    if zonemap is not None:
        zonemap_file = Path("zonemap.txt")
        zonemap_file.write_text(zonemap, encoding="utf-8")
    else:
        zonemap_file = None
    rft_config = RFTConfig(input_files=["DUMMY"], zonemap=zonemap_file)
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
def test_that_get_rft_observations_is_active_based_on_matching_pressure_response():
    """Test that is_active is True when there is a matching PRESSURE response,
    and False if not"""
    observations = [
        _create_rft_observation(),
        _create_rft_observation(
            obs_name="obs2",
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
            obs_name="obs2",
            tvd=30.0,
            md=60.0,
            value=160.0,
        ),
        _create_rft_observation(
            obs_name="obs3",
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
        _create_rft_observation(obs_name="obs1", tvd=25.0, md=50.0),
        _create_rft_observation(obs_name="obs2", tvd=30.0, md=60.0),
        _create_rft_observation(obs_name="obs3", tvd=35.0, md=70.0),
        _create_rft_observation(well="WELL2", obs_name="obs4", tvd=40.0, md=80.0),
        _create_rft_observation(well="WELL2", obs_name="obs5", tvd=45.0, md=90.0),
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
            obs_name="obs2",
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
def test_that_rft_artifacts_are_saved_atomically():
    def read_from_file_mock(run_path, iens, iter_):
        if iens in {0, 2}:
            return _create_rft_response_df()
        elif iens == 1:
            raise InvalidResponseFile("Mock error for realization 1")

    def location_metadata_mock(run_path, iens, iter_, observations):
        if iens in {0, 1}:
            return _create_rft_location_metadata_df()
        elif iens == 2:
            raise InvalidResponseFile("Mock error for realization 2")

    async def run_test():
        with (
            patch.object(RFTConfig, "read_from_file", side_effect=read_from_file_mock),
            patch.object(
                RFTConfig,
                "obtain_location_metadata",
                side_effect=location_metadata_mock,
            ),
        ):
            num_realizations = 3
            observations = [
                _create_rft_observation(),
            ]

            with _create_rft_ensemble(num_realizations, observations) as ensemble:
                await _write_responses_to_storage("", 0, ensemble)
                await _write_responses_to_storage("", 1, ensemble)
                await _write_responses_to_storage("", 2, ensemble)

                ensemble.load_responses("rft", (0,))
                with pytest.raises(KeyError):
                    ensemble.load_responses("rft", (1,))

                with pytest.raises(KeyError):
                    ensemble.load_responses("rft", (2,))

                ensemble.load_observation_location_metadata(0)
                with pytest.raises(FileNotFoundError):
                    ensemble.load_observation_location_metadata(1)
                with pytest.raises(FileNotFoundError):
                    ensemble.load_observation_location_metadata(2)

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


def test_that_save_transition_data_writes_file_to_disk(tmp_path):
    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment()
        ensemble = storage.create_ensemble(
            experiment, ensemble_size=1, iteration=0, name="prior"
        )

        ensemble.save_transition_data("report.json", '{"key": "value"}')

        written = (ensemble._path / "transition" / "report.json").read_text(
            encoding="utf-8"
        )
        assert written == '{"key": "value"}'


def test_that_save_transition_data_creates_transition_directory(tmp_path):
    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment()
        ensemble = storage.create_ensemble(
            experiment, ensemble_size=1, iteration=0, name="prior"
        )

        transition_dir = ensemble._path / "transition"
        assert not transition_dir.exists()

        ensemble.save_transition_data("report.json", "data")
        assert transition_dir.is_dir()


def test_that_save_transition_data_raises_in_read_mode(tmp_path):
    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment()
        storage.create_ensemble(experiment, ensemble_size=1, iteration=0, name="prior")

    with open_storage(tmp_path, mode="r") as storage:
        ensemble = next(iter(storage.ensembles))
        with pytest.raises(ModeError):
            ensemble.save_transition_data("report.json", "data")
