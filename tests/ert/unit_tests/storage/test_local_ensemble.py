import numpy as np
import polars as pl
import pytest

from ert.config import GenKwConfig, RFTConfig
from ert.exceptions import StorageError
from ert.storage import open_storage


def test_that_load_scalar_keys_loads_all_parameters(tmp_path):
    """Test that load_scalar_keys loads all scalar parameters when keys=None."""
    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment(
            parameters=[
                GenKwConfig(
                    name="param1",
                    group="group1",
                    distribution={"name": "uniform", "min": 0, "max": 1},
                ),
                GenKwConfig(
                    name="param2",
                    group="group1",
                    distribution={"name": "uniform", "min": 0, "max": 1},
                ),
                GenKwConfig(
                    name="param3",
                    group="group2",
                    distribution={"name": "normal", "mean": 0, "std": 1},
                ),
            ]
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
            parameters=[
                GenKwConfig(
                    name="param1",
                    group="group1",
                    distribution={"name": "uniform", "min": 0, "max": 1},
                ),
                GenKwConfig(
                    name="param2",
                    group="group1",
                    distribution={"name": "uniform", "min": 0, "max": 1},
                ),
                GenKwConfig(
                    name="param3",
                    group="group2",
                    distribution={"name": "normal", "mean": 0, "std": 1},
                ),
            ]
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
            parameters=[
                GenKwConfig(
                    name="param1",
                    group="group1",
                    distribution={"name": "uniform", "min": 0, "max": 1},
                ),
            ]
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
            parameters=[
                GenKwConfig(
                    name="param1",
                    group="group1",
                    distribution={"name": "uniform", "min": 0, "max": 1},
                ),
            ]
        )
        ensemble = storage.create_ensemble(experiment.id, ensemble_size=5, name="test")

        with pytest.raises(KeyError, match="No SCALAR dataset in storage"):
            ensemble.load_scalar_keys(keys=["param1"])


def test_that_load_scalar_keys_raises_key_error_for_unregistered_parameters(tmp_path):
    """Test that load_scalar_keys raises KeyError for parameters not in experiment."""
    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment(
            parameters=[
                GenKwConfig(
                    name="param1",
                    group="group1",
                    distribution={"name": "uniform", "min": 0, "max": 1},
                ),
            ]
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
            parameters=[
                GenKwConfig(
                    name="param1",
                    group="group1",
                    distribution={"name": "uniform", "min": 0, "max": 1},
                ),
            ]
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


def _create_rft_observation_df(
    well: str,
    date: str,
    prop: str,
    obs_name: str,
    east: float,
    north: float,
    tvd: float,
    md: float | None,
    zone: str | None,
    observations: float,
    std: float,
) -> pl.DataFrame:
    """Helper to create a single RFT observation row."""
    return pl.DataFrame(
        {
            "response_key": [f"{well}:{date}:{prop}"],
            "observation_key": [obs_name],
            "east": pl.Series([east], dtype=pl.Float32),
            "north": pl.Series([north], dtype=pl.Float32),
            "tvd": pl.Series([tvd], dtype=pl.Float32),
            "md": pl.Series([md], dtype=pl.Float32),
            "zone": pl.Series([zone], dtype=pl.String),
            "observations": pl.Series([observations], dtype=pl.Float32),
            "std": pl.Series([std], dtype=pl.Float32),
            "radius": pl.Series([None], dtype=pl.Float32),
        }
    )


def _create_rft_response_df(
    well: str,
    date: str,
    prop: str,
    value: float,
    east: float,
    north: float,
    tvd: float,
    zone: str | None,
    i: int = 0,
    j: int = 0,
    k: int = 0,
) -> pl.DataFrame:
    """Helper to create a single RFT response row."""
    return pl.DataFrame(
        {
            "response_key": [f"{well}:{date}:{prop}"],
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


def test_that_get_rft_observations_and_responses_returns_joined_data(tmp_path):
    rft_config = RFTConfig(input_files=["DUMMY"])

    observations = pl.concat(
        [
            _create_rft_observation_df(
                "WELL1",
                "2020-01-01",
                "PRESSURE",
                "obs1",
                100.0,
                200.0,
                50.0,
                25.0,
                "Z1",
                150.0,
                5.0,
            ),
        ]
    )

    responses_real0 = pl.concat(
        [
            _create_rft_response_df(
                "WELL1", "2020-01-01", "PRESSURE", 148.0, 100.0, 200.0, 50.0, "Z1"
            ),
            _create_rft_response_df(
                "WELL1", "2020-01-01", "SGAS", 0.1, 100.0, 200.0, 50.0, "Z1"
            ),
            _create_rft_response_df(
                "WELL1", "2020-01-01", "SWAT", 0.2, 100.0, 200.0, 50.0, "Z1"
            ),
        ]
    )

    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment(
            responses=[rft_config],
            observations={"rft": observations},
        )
        ensemble = storage.create_ensemble(experiment.id, ensemble_size=1, name="test")
        ensemble.save_response("rft", responses_real0, 0)

        result = ensemble.get_rft_observations_and_responses()

        assert result.shape[0] == 1
        assert "utm_x" in result.columns
        assert "utm_y" in result.columns
        assert "true_vertical_depth" in result.columns
        assert "measured_depth" in result.columns
        assert "time" in result.columns
        assert "report_step" in result.columns
        assert "observed" in result.columns
        assert "error" in result.columns
        assert result["utm_x"][0] == 100.0
        assert result["utm_y"][0] == 200.0
        assert result["pressure"][0] == pytest.approx(148.0)
        assert result["observed"][0] == pytest.approx(150.0)
        assert result["well"][0] == "WELL1"


def test_that_get_rft_observations_and_responses_computes_soil(tmp_path):
    """Test that SOIL is computed as 1 - SGAS - SWAT."""
    rft_config = RFTConfig(input_files=["DUMMY"])

    observations = _create_rft_observation_df(
        "WELL1",
        "2020-01-01",
        "PRESSURE",
        "obs1",
        100.0,
        200.0,
        50.0,
        25.0,
        None,
        150.0,
        5.0,
    )

    responses_real0 = pl.concat(
        [
            _create_rft_response_df(
                "WELL1", "2020-01-01", "PRESSURE", 148.0, 100.0, 200.0, 50.0, None
            ),
            _create_rft_response_df(
                "WELL1", "2020-01-01", "SGAS", 0.15, 100.0, 200.0, 50.0, None
            ),
            _create_rft_response_df(
                "WELL1", "2020-01-01", "SWAT", 0.35, 100.0, 200.0, 50.0, None
            ),
        ]
    )

    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment(
            responses=[rft_config],
            observations={"rft": observations},
        )
        ensemble = storage.create_ensemble(experiment.id, ensemble_size=1, name="test")
        ensemble.save_response("rft", responses_real0, 0)

        result = ensemble.get_rft_observations_and_responses()

        assert result["soil"][0] == pytest.approx(0.5)  # 1 - 0.15 - 0.35 = 0.5


def test_that_get_rft_observations_and_responses_sets_is_active_based_on_pressure(
    tmp_path,
):
    """Test that is_active is True when PRESSURE there is a matching response,
    and False if not"""
    rft_config = RFTConfig(input_files=["DUMMY"])

    observations = pl.concat(
        [
            _create_rft_observation_df(
                "WELL1",
                "2020-01-01",
                "PRESSURE",
                "obs1",
                100.0,
                200.0,
                50.0,
                25.0,
                None,
                150.0,
                5.0,
            ),
            _create_rft_observation_df(
                "WELL1",
                "2020-01-01",
                "PRESSURE",
                "obs2",
                100.0,
                200.0,
                60.0,
                30.0,
                None,
                160.0,
                5.0,
            ),
        ]
    )

    responses_real0 = pl.concat(
        [
            _create_rft_response_df(
                "WELL1", "2020-01-01", "PRESSURE", 148.0, 100.0, 200.0, 50.0, None
            ),
            _create_rft_response_df(
                "WELL1", "2020-01-01", "SGAS", 0.1, 100.0, 200.0, 50.0, None
            ),
        ]
    )

    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment(
            responses=[rft_config],
            observations={"rft": observations},
        )
        ensemble = storage.create_ensemble(experiment.id, ensemble_size=1, name="test")
        ensemble.save_response("rft", responses_real0, 0)

        result = ensemble.get_rft_observations_and_responses().sort(
            "true_vertical_depth"
        )

        assert result["is_active"][0] is True  # tvd=50 has pressure
        assert (
            result["is_active"][1] is False
        )  # tvd=60 has no matching response (left join)


def test_that_get_rft_observations_and_responses_sets_valid_zone_with_null_equality(
    tmp_path,
):
    """Test that valid_zone is True when zone equals response_zone,
    including None == None."""
    rft_config = RFTConfig(input_files=["DUMMY"])

    observations = pl.concat(
        [
            _create_rft_observation_df(
                "WELL1",
                "2020-01-01",
                "PRESSURE",
                "obs1",
                100.0,
                200.0,
                50.0,
                25.0,
                "Z1",
                150.0,
                5.0,
            ),
            _create_rft_observation_df(
                "WELL1",
                "2020-01-01",
                "PRESSURE",
                "obs2",
                100.0,
                200.0,
                60.0,
                30.0,
                None,
                160.0,
                5.0,
            ),
            _create_rft_observation_df(
                "WELL1",
                "2020-01-01",
                "PRESSURE",
                "obs3",
                100.0,
                200.0,
                70.0,
                35.0,
                "Z2",
                170.0,
                5.0,
            ),
        ]
    )

    responses_real0 = pl.concat(
        [
            _create_rft_response_df(
                "WELL1", "2020-01-01", "PRESSURE", 148.0, 100.0, 200.0, 50.0, "Z1"
            ),
            _create_rft_response_df(
                "WELL1", "2020-01-01", "PRESSURE", 158.0, 100.0, 200.0, 60.0, None
            ),
            _create_rft_response_df(
                "WELL1", "2020-01-01", "PRESSURE", 168.0, 100.0, 200.0, 70.0, "Z1"
            ),
        ]
    )

    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment(
            responses=[rft_config],
            observations={"rft": observations},
        )
        ensemble = storage.create_ensemble(experiment.id, ensemble_size=1, name="test")
        ensemble.save_response("rft", responses_real0, 0)

        result = ensemble.get_rft_observations_and_responses().sort(
            "true_vertical_depth"
        )

        assert result["valid_zone"][0] is True  # Z1 == Z1
        assert result["valid_zone"][1] is True  # None == None (eq_missing)
        assert result["valid_zone"][2] is False  # Z2 != Z1


def test_that_get_rft_observations_and_responses_handles_multiple_realizations(
    tmp_path,
):
    """Test that responses from multiple realizations are concatenated correctly."""
    rft_config = RFTConfig(input_files=["DUMMY"])

    observations = _create_rft_observation_df(
        "WELL1",
        "2020-01-01",
        "PRESSURE",
        "obs1",
        100.0,
        200.0,
        50.0,
        25.0,
        None,
        150.0,
        5.0,
    )

    responses_real0 = _create_rft_response_df(
        "WELL1", "2020-01-01", "PRESSURE", 148.0, 100.0, 200.0, 50.0, None
    )
    responses_real1 = _create_rft_response_df(
        "WELL1", "2020-01-01", "PRESSURE", 152.0, 100.0, 200.0, 50.0, None
    )

    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment(
            responses=[rft_config],
            observations={"rft": observations},
        )
        ensemble = storage.create_ensemble(experiment.id, ensemble_size=2, name="test")
        ensemble.save_response("rft", responses_real0, 0)
        ensemble.save_response("rft", responses_real1, 1)

        result = ensemble.get_rft_observations_and_responses().sort("report_step")

        assert result.shape[0] == 2
        assert result["report_step"].to_list() == [0, 1]
        assert result["pressure"][0] == pytest.approx(148.0)
        assert result["pressure"][1] == pytest.approx(152.0)


def test_that_get_rft_observations_and_responses_raises_error_for_no_observations(
    tmp_path,
):
    """Test that StorageError is raised when no RFT observations exist."""
    rft_config = RFTConfig(input_files=["DUMMY"])

    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment(
            responses=[rft_config],
            observations={},
        )
        ensemble = storage.create_ensemble(experiment.id, ensemble_size=1, name="test")

        with pytest.raises(StorageError, match="No RFT observations found"):
            ensemble.get_rft_observations_and_responses()


def test_that_get_rft_observations_and_responses_raises_error_for_no_rft_config(
    tmp_path,
):
    """Test that KeyError is raised when no RFT response configuration exists."""
    observations = _create_rft_observation_df(
        "WELL1",
        "2020-01-01",
        "PRESSURE",
        "obs1",
        100.0,
        200.0,
        50.0,
        25.0,
        None,
        150.0,
        5.0,
    )

    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment(
            responses=[],
            observations={"rft": observations},
        )
        ensemble = storage.create_ensemble(experiment.id, ensemble_size=1, name="test")

        with pytest.raises(KeyError, match="No RFT response configuration found"):
            ensemble.get_rft_observations_and_responses()


def test_that_get_rft_observations_and_responses_raises_error_when_response_not_saved(
    tmp_path,
):
    rft_config = RFTConfig(input_files=["DUMMY"])

    observations = _create_rft_observation_df(
        "WELL1",
        "2020-01-01",
        "PRESSURE",
        "obs1",
        100.0,
        200.0,
        50.0,
        25.0,
        None,
        150.0,
        5.0,
    )

    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment(
            responses=[rft_config],
            observations={"rft": observations},
        )
        ensemble = storage.create_ensemble(experiment.id, ensemble_size=1, name="test")

        with pytest.raises(KeyError, match="No response for key rft"):
            ensemble.get_rft_observations_and_responses()


def test_that_get_rft_observations_and_responses_adds_missing_saturation_columns(
    tmp_path,
):
    rft_config = RFTConfig(input_files=["DUMMY"])

    observations = _create_rft_observation_df(
        "WELL1",
        "2020-01-01",
        "PRESSURE",
        "obs1",
        100.0,
        200.0,
        50.0,
        25.0,
        None,
        150.0,
        5.0,
    )

    responses_real0 = _create_rft_response_df(
        "WELL1", "2020-01-01", "PRESSURE", 148.0, 100.0, 200.0, 50.0, None
    )

    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment(
            responses=[rft_config],
            observations={"rft": observations},
        )
        ensemble = storage.create_ensemble(experiment.id, ensemble_size=1, name="test")
        ensemble.save_response("rft", responses_real0, 0)

        result = ensemble.get_rft_observations_and_responses()

        assert "sgas" in result.columns
        assert "swat" in result.columns
        assert result["sgas"][0] is None
        assert result["swat"][0] is None
        assert result["soil"][0] is None  # 1 - None - None = None


def test_that_get_rft_observations_and_responses_extracts_well_and_date_from_response_key(  # noqa: E501
    tmp_path,
):
    rft_config = RFTConfig(input_files=["DUMMY"])

    observations = _create_rft_observation_df(
        "PRODUCER_A",
        "2021-06-15",
        "PRESSURE",
        "obs1",
        100.0,
        200.0,
        50.0,
        25.0,
        None,
        150.0,
        5.0,
    )

    responses_real0 = _create_rft_response_df(
        "PRODUCER_A", "2021-06-15", "PRESSURE", 148.0, 100.0, 200.0, 50.0, None
    )

    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment(
            responses=[rft_config],
            observations={"rft": observations},
        )
        ensemble = storage.create_ensemble(experiment.id, ensemble_size=1, name="test")
        ensemble.save_response("rft", responses_real0, 0)

        result = ensemble.get_rft_observations_and_responses()

        assert result["well"][0] == "PRODUCER_A"
        assert result["time"][0] == "2021-06-15"


def test_that_get_rft_observations_and_responses_converts_property_to_lowercase(
    tmp_path,
):
    rft_config = RFTConfig(input_files=["DUMMY"])

    observations = _create_rft_observation_df(
        "WELL1",
        "2020-01-01",
        "PRESSURE",
        "obs1",
        100.0,
        200.0,
        50.0,
        25.0,
        None,
        150.0,
        5.0,
    )

    responses_real0 = pl.concat(
        [
            _create_rft_response_df(
                "WELL1", "2020-01-01", "PRESSURE", 148.0, 100.0, 200.0, 50.0, None
            ),
            _create_rft_response_df(
                "WELL1", "2020-01-01", "SGAS", 0.1, 100.0, 200.0, 50.0, None
            ),
            _create_rft_response_df(
                "WELL1", "2020-01-01", "SWAT", 0.2, 100.0, 200.0, 50.0, None
            ),
        ]
    )

    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment(
            responses=[rft_config],
            observations={"rft": observations},
        )
        ensemble = storage.create_ensemble(experiment.id, ensemble_size=1, name="test")
        ensemble.save_response("rft", responses_real0, 0)

        result = ensemble.get_rft_observations_and_responses()

        assert "pressure" in result.columns
        assert "sgas" in result.columns
        assert "swat" in result.columns
        assert "PRESSURE" not in result.columns
        assert "SGAS" not in result.columns
        assert "SWAT" not in result.columns
