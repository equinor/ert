from io import BytesIO, StringIO
from textwrap import dedent
from typing import cast

import polars as pl
import pytest

from ert.config._create_observation_dataframes import _handle_seismic_observation
from ert.config._observations import SeismicObservation
from ert.config.ert_config import ErtConfig
from ert.config.response_config import InvalidResponseFile
from ert.config.seismic_config import SeismicConfig
from tests.ert.defaults_generator import create_seismic_observation_dict


def _mock_seismic_response(
    mocked_files: dict, path: str, frame: pl.DataFrame, suffix: str
) -> None:
    """Serialize ``frame`` into ``mocked_files[path]`` matching ``suffix``.

    Uses a StringIO for CSV and a BytesIO for parquet so the same DataFrame
    is the source of truth for both file formats consumed by
    ``SeismicConfig.read_from_file``.
    """
    if suffix == ".parquet":
        buf = BytesIO()
        frame.write_parquet(buf)
        mocked_files[path] = buf.getvalue()
    else:
        buf = StringIO()
        frame.write_csv(buf)
        mocked_files[path] = buf.getvalue()


def test_that_seismic_observation_response_key_matches_simulated_response_key(
    mocked_files,
):
    expected_response_key = "horizon--amplitude_full_min_depth--20250101_20240101"
    name = f"{expected_response_key}.csv"
    runpath = "/runpath"
    obs_path = "share/preprocessed/tables/" + name
    simulated_path_relative_to_runpath = "share/results/tables/" + name
    simulated_path = runpath + "/" + simulated_path_relative_to_runpath

    mocked_files[obs_path] = dedent(
        """
        X_UTME,Y_UTMN,OBS,OBS_ERROR,REGION
        100.00,200.00,1.0,0.005,1.0
        105.00,205.00,2.0,0.005,1.0
        """
    )

    mocked_files[simulated_path] = dedent(
        """
        X_UTME,Y_UTMN,OBS,OBS_ERROR,REGION
        100.00,200.00,1.1,0.005,1.0
        105.00,205.00,2.2,0.005,1.0
        """
    )

    config = ErtConfig.from_dict(
        {
            "SEISMIC": [simulated_path_relative_to_runpath],
            "OBS_CONFIG": (
                "obsconf",
                [
                    create_seismic_observation_dict(csv=obs_path),
                ],
            ),
        }
    )

    seismic_config = cast(
        SeismicConfig, config.ensemble_config.response_configs["seismic"]
    )
    observation = cast(SeismicObservation, config.observation_declarations[0])
    observations = _handle_seismic_observation(observation)
    data = seismic_config.read_from_file(runpath, 1, 1)
    assert "response_key" in data.columns
    assert "response_key" in observations.columns
    assert set(data["response_key"].unique()) == set(
        observations["response_key"].unique()
    )
    assert set(data["response_key"].unique()) == {expected_response_key}


def test_that_seismic_config_raises_when_reading_from_non_existing_file(tmp_path):
    seismic_config = SeismicConfig(
        input_files=["non-existent-file.csv"],
        keys=["key"],
    )
    with pytest.raises(InvalidResponseFile):
        seismic_config.read_from_file(tmp_path / "non-existent-file.csv", 1, 1)


@pytest.mark.parametrize("suffix", [".csv", ".parquet"])
def test_that_seismic_config_reads_from_all_input_files(mocked_files, suffix):
    key1 = "horizon--amplitude_full_min_depth--20250101_20240101"
    key2 = "horizon--amplitude_full_mean_depth--20260101_20240101"
    name1 = f"{key1}{suffix}"
    name2 = f"{key2}{suffix}"
    runpath = "/runpath"

    frame1 = pl.DataFrame(
        {
            "X_UTME": [100.0, 105.0],
            "Y_UTMN": [200.0, 205.0],
            "OBS": [1.0, 2.0],
            "OBS_ERROR": [0.005, 0.005],
            "REGION": [1.0, 1.0],
        }
    )
    frame2 = pl.DataFrame(
        {
            "X_UTME": [100.0, 105.0],
            "Y_UTMN": [200.0, 205.0],
            "OBS": [3.0, 4.0],
            "OBS_ERROR": [0.005, 0.005],
            "REGION": [1.0, 1.0],
        }
    )
    _mock_seismic_response(mocked_files, f"{runpath}/{name1}", frame1, suffix)
    _mock_seismic_response(mocked_files, f"{runpath}/{name2}", frame2, suffix)

    seismic_config = SeismicConfig(
        input_files=[name1, name2],
        keys=[key1, key2],
    )

    data = seismic_config.read_from_file(runpath, 1, 1)
    assert data.shape == (4, 4)
    assert data["response_key"].to_list() == [key1, key1, key2, key2]
    assert data["east"].to_list() == [100.0, 105.0, 100.0, 105.0]
    assert data["north"].to_list() == [200.0, 205.0, 200.0, 205.0]
    assert data["values"].to_list() == [1.0, 2.0, 3.0, 4.0]


@pytest.mark.parametrize("suffix", [".csv", ".parquet"])
def test_that_empty_seismic_response_file_does_not_raise(mocked_files, suffix):
    key = "horizon--amplitude_full_min_depth--20250101_20240101"
    name = f"{key}{suffix}"
    runpath = "/runpath"

    empty = pl.DataFrame(
        schema={
            "X_UTME": pl.Float64,
            "Y_UTMN": pl.Float64,
            "OBS": pl.Float64,
            "OBS_ERROR": pl.Float64,
            "REGION": pl.Float64,
        }
    )
    _mock_seismic_response(mocked_files, f"{runpath}/{name}", empty, suffix)

    seismic_config = SeismicConfig(
        input_files=[name],
        keys=[key],
    )

    data = seismic_config.read_from_file(runpath, 1, 1)
    assert data.is_empty()


@pytest.mark.parametrize(
    ("east", "north"),
    [
        pytest.param([111.25, 111.25], [222.25, 222.25], id="same coordinates"),
        pytest.param([0.0, 0.0], [0.1953125, 0.0], id="less than double tolerance"),
    ],
)
@pytest.mark.parametrize("suffix", [".csv", ".parquet"])
def test_that_seismic_response_coordinate_distance_below_tolerance_raises(
    mocked_files, east, north, suffix
):
    key = "horizon--amplitude_full_min_depth--20250101_20240101"
    name = f"{key}{suffix}"
    runpath = "/runpath"

    frame = pl.DataFrame(
        {
            "X_UTME": east,
            "Y_UTMN": north,
            "OBS": [1.0, 2.0],
            "OBS_ERROR": [0.005, 0.005],
            "REGION": [1.0, 1.0],
        }
    )
    _mock_seismic_response(mocked_files, f"{runpath}/{name}", frame, suffix)

    seismic_config = SeismicConfig(
        input_files=[name],
        keys=[key],
    )

    with pytest.raises(InvalidResponseFile) as err:
        seismic_config.read_from_file(runpath, 1, 1)

    assert (
        "Seismic response coordinates with approximate locations "
        f"[(({east[0]}, {north[0]}), ({east[1]}, {north[1]}))] "
        "fall inside of a tolerance radius." in str(err.value)
    )
