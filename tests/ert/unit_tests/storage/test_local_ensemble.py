from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import hypothesis.strategies as st
import numpy as np
import polars as pl
import pytest
from hypothesis import given

from ert.config import GenKwConfig, RFTConfig, SummaryConfig
from ert.config._observations import RFTObservation
from ert.exceptions import StorageError
from ert.storage import open_storage


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
        radius=None,
        tvd=tvd,
        md=md,
        zone=zone,
    )


def _create_rft_response_df(
    *,
    well: str = "WELL1",
    date: str = "2020-01-01",
    prop: str = "PRESSURE",
    value: float = 148.0,
    east: float = 100.0,
    north: float = 200.0,
    tvd: float = 25.0,
    zone: str | None = None,
    i: int = 1,
    j: int = 2,
    k: int = 3,
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "response_key": [f"{well}:{date}:{prop}"],
            "well": [well],
            "date": [date],
            "property": [prop],
            "values": pl.Series([value], dtype=pl.Float32),
            "east": pl.Series([east], dtype=pl.Float32),
            "north": pl.Series([north], dtype=pl.Float32),
            "tvd": pl.Series([tvd], dtype=pl.Float32),
            "zone": pl.Series([zone], dtype=pl.String),
            "i": pl.Series([i], dtype=pl.Int32),
            "j": pl.Series([j], dtype=pl.Int32),
            "k": pl.Series([k], dtype=pl.Int32),
        }
    )


@contextmanager
def _create_rft_ensemble(ensemble_size, observations, zonemap=None):
    if zonemap is not None:
        zonemap_file = Path("zonemap.txt")
        zonemap_file.write_text(zonemap, encoding="utf-8")
    else:
        zonemap_file = None
    rft_config = RFTConfig(input_files=["DUMMY"], zonemap=zonemap_file)
    with open_storage("storage", mode="w") as storage:
        experiment = storage.create_experiment(
            experiment_config={
                "response_configuration": [rft_config.model_dump(mode="json")],
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
        ensemble.save_response(
            "rft",
            pl.concat(
                [
                    _create_rft_response_df(zone="Z1", prop="PRESSURE", value=148.0),
                    _create_rft_response_df(zone="Z1", prop="SGAS", value=0.1),
                    _create_rft_response_df(zone="Z1", prop="SWAT", value=0.2),
                ]
            ),
            0,
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
        ensemble.save_response("rft", pl.concat([_create_rft_response_df()]), 0)

        result = ensemble.get_rft_observations_and_responses().sort(
            "true_vertical_depth"
        )

        assert result["is_active"][0] is True  # tvd=50 has pressure response
        assert result["is_active"][1] is False  # tvd=60 has no pressure response


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

    with _create_rft_ensemble(1, observations, zonemap="1 Z1\n2 Z2\n") as ensemble:
        ensemble.save_response(
            "rft",
            pl.concat(
                [
                    _create_rft_response_df(zone="Z1", tvd=25.0),
                    _create_rft_response_df(zone=None, tvd=30.0),
                    _create_rft_response_df(zone="Z1", tvd=35.0),
                ]
            ),
            0,
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
        ensemble.save_response("rft", pl.concat([_create_rft_response_df()]), 0)

        result = ensemble.get_rft_observations_and_responses().sort(
            ["well", "true_vertical_depth"]
        )

        assert result["well"].to_list() == ["WELL1", "WELL1", "WELL1", "WELL2", "WELL2"]
        assert result["order"].to_list() == [0, 1, 2, 0, 1]


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.integration_test
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
def test_that_get_rft_observations_and_responses_adds_missing_saturation_columns():
    with _create_rft_ensemble(1, [_create_rft_observation()]) as ensemble:
        # Save a response that does not contain saturations
        ensemble.save_response("rft", _create_rft_response_df(), 0)

        result = ensemble.get_rft_observations_and_responses()

        assert "sgas" in result.columns
        assert "swat" in result.columns
        assert "soil" in result.columns
        assert result["sgas"][0] is None
        assert result["swat"][0] is None
        assert result["soil"][0] is None


def test_that_get_rft_observations_and_responses_maps_report_step_from_summary_times(
    tmp_path,
):
    rft_config = RFTConfig(input_files=["DUMMY"])
    summary_config = SummaryConfig(keys=["FOPR"], input_files=["DUMMY"])

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
                [datetime(2020, 1, 1), datetime(2020, 1, 15), datetime(2020, 2, 15)]
            ).dt.cast_time_unit("ms"),
            "values": pl.Series([100.0, 200.0, 300.0], dtype=pl.Float32),
        }
    )

    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment(
            experiment_config={
                "response_configuration": [
                    rft_config.model_dump(mode="json"),
                    summary_config.model_dump(mode="json"),
                ],
                "observations": [o.model_dump(mode="json") for o in observations],
            }
        )
        ensemble = storage.create_ensemble(experiment.id, ensemble_size=1, name="test")
        ensemble.save_response("rft", rft_responses, 0)
        ensemble.save_response("summary", summary_responses, 0)

        result = ensemble.get_rft_observations_and_responses().sort("time")

        assert "report_step" in result.columns
        assert result["report_step"][0] == 1
        assert result["report_step"][1] == 2
