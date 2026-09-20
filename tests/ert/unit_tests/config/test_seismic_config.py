from io import BytesIO, StringIO
from pathlib import Path
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


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.parametrize("suffix", [".csv", ".parquet"])
def test_that_seismic_observation_response_key_matches_simulated_response_key(
    mocked_files, suffix
):
    expected_response_key = "horizon--amplitude_full_min_depth--20250101_20240101"
    name = f"{expected_response_key}{suffix}"
    runpath = "runpath"
    obs_path = "share/preprocessed/tables/" + name
    simulated_path_relative_to_runpath = "share/results/tables/" + name
    simulated_path = runpath + "/" + simulated_path_relative_to_runpath

    obs_frame = pl.DataFrame(
        {
            "X_UTME": [100.0, 105.0],
            "Y_UTMN": [200.0, 205.0],
            "OBS": [1.0, 2.0],
            "OBS_ERROR": [0.005, 0.005],
            "REGION": [1.0, 1.0],
        }
    )

    simulated_frame = pl.DataFrame(
        {
            "X_UTME": [100.0, 105.0],
            "Y_UTMN": [200.0, 205.0],
            "OBS": [1.1, 2.2],
            "OBS_ERROR": [0.005, 0.005],
            "REGION": [1.0, 1.0],
        }
    )

    _mock_seismic_response(mocked_files, obs_path, obs_frame, suffix)
    _mock_seismic_response(mocked_files, simulated_path, simulated_frame, suffix)

    Path(obs_path).parent.mkdir(parents=True, exist_ok=True)
    Path(simulated_path).parent.mkdir(parents=True, exist_ok=True)

    if suffix.endswith("parquet"):
        Path(obs_path).write_bytes(mocked_files[obs_path])
        Path(simulated_path).write_bytes(mocked_files[simulated_path])
    else:
        Path(obs_path).write_text(mocked_files[obs_path], encoding="utf8")
        Path(simulated_path).write_text(mocked_files[simulated_path], encoding="utf8")

    config = ErtConfig.from_dict(
        {
            "SEISMIC": [simulated_path_relative_to_runpath],
            "OBS_CONFIG": (
                "obsconf",
                [
                    create_seismic_observation_dict(obs_file=obs_path),
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


def test_that_unsupported_seismic_response_file_extension_raises_invalid_response_file(
    mocked_files,
):
    key = "horizon--amplitude_full_min_depth--20250101_20240101"
    name = f"{key}.txt"
    mocked_files[f"{name}"] = "irrelevant content"

    seismic_config = SeismicConfig(
        input_files=[name],
        keys=[key],
    )

    with pytest.raises(InvalidResponseFile) as err:
        seismic_config.read_from_file("", 1, 1)

    assert (
        f"Unsupported seismic response file extension '.txt' for {name}. "
        "Expected '.csv' or '.parquet'." in str(err.value)
    )


def test_that_seismic_config_raises_when_reading_from_non_existing_file(tmp_path):
    seismic_config = SeismicConfig(
        input_files=["non-existent-file.csv"],
        keys=["key"],
    )
    with pytest.raises(InvalidResponseFile):
        seismic_config.read_from_file(tmp_path, 1, 1)


@pytest.mark.parametrize("suffix", [".csv", ".parquet"])
def test_that_seismic_config_reads_from_all_input_files(mocked_files, suffix):
    key1 = "horizon--amplitude_full_min_depth--20250101_20240101"
    key2 = "horizon--amplitude_full_mean_depth--20260101_20240101"
    name1 = f"{key1}{suffix}"
    name2 = f"{key2}{suffix}"

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
    _mock_seismic_response(mocked_files, f"{name1}", frame1, suffix)
    _mock_seismic_response(mocked_files, f"{name2}", frame2, suffix)

    seismic_config = SeismicConfig(
        input_files=[name1, name2],
        keys=[key1, key2],
    )

    data = seismic_config.read_from_file("", 1, 1)
    assert data.shape == (4, 4)
    assert data["response_key"].to_list() == [key1, key1, key2, key2]
    assert data["east"].to_list() == [100.0, 105.0, 100.0, 105.0]
    assert data["north"].to_list() == [200.0, 205.0, 200.0, 205.0]
    assert data["values"].to_list() == [1.0, 2.0, 3.0, 4.0]


@pytest.mark.usefixtures("use_tmpdir")
def test_that_seismic_config_supports_glob_pattern():
    key1 = "horizon1--amplitude_full_mean_depth--20260101_20240101"
    key2 = "horizon1--amplitude_full_min_depth--20250101_20240101"
    key3 = "horizon2--amplitude_far_min_depth--20250101_20240101"
    key4 = "other_horizon--amplitude_full_mean_depth--20260101_20240101"

    runpath = "runpath"
    Path(runpath).mkdir(parents=True, exist_ok=True)

    for key in [key1, key2, key3, key4]:
        name = f"{key}.csv"
        simulated_path = runpath + "/" + name
        content = dedent(
            """
            X_UTME,Y_UTMN,OBS,OBS_ERROR,REGION
            100.00,200.00,1.0,0.005,1.0
            """
        )
        Path(simulated_path).write_text(
            content,
            encoding="utf8",
        )

    pattern1 = "horizon1--amplitude_full*"
    pattern2 = "horizon2--amplitude_far*"
    duplicate_pattern = "horizon*"
    config = ErtConfig.from_dict(
        {
            "SEISMIC": [pattern1, pattern2, duplicate_pattern],
        }
    )

    seismic_config = cast(
        SeismicConfig, config.ensemble_config.response_configs["seismic"]
    )

    data = seismic_config.read_from_file(runpath, 1, 1)
    assert sorted(data["response_key"].to_list()) == [key1, key2, key3]


@pytest.mark.parametrize("suffix", [".csv", ".parquet"])
def test_that_empty_seismic_response_file_does_not_raise(mocked_files, suffix):
    key = "horizon--amplitude_full_min_depth--20250101_20240101"
    name = f"{key}{suffix}"

    empty = pl.DataFrame(
        schema={
            "X_UTME": pl.Float64,
            "Y_UTMN": pl.Float64,
            "OBS": pl.Float64,
            "OBS_ERROR": pl.Float64,
            "REGION": pl.Float64,
        }
    )
    _mock_seismic_response(mocked_files, f"{name}", empty, suffix)

    seismic_config = SeismicConfig(
        input_files=[name],
        keys=[key],
    )

    data = seismic_config.read_from_file("", 1, 1)
    assert data.is_empty()


@pytest.mark.parametrize(
    ("east", "north"),
    [
        pytest.param([111.25, 111.25], [222.25, 222.25], id="same coordinates"),
        pytest.param([0.0, 0.0], [0.1953125, 0.0], id="less than double tolerance"),
    ],
)
def test_that_seismic_response_coordinate_distance_below_tolerance_raises(
    mocked_files, east, north
):
    key = "horizon--amplitude_full_min_depth--20250101_20240101"
    name = f"{key}.csv"

    mocked_files[name] = dedent(
        f"""
        X_UTME,Y_UTMN,OBS,OBS_ERROR,REGION
        {east[0]},{north[0]},1.0,0.005,1.0
        {east[1]},{north[1]},2.0,0.005,1.0
        """
    )

    seismic_config = SeismicConfig(
        input_files=[name],
        keys=[key],
    )

    with pytest.raises(InvalidResponseFile) as err:
        seismic_config.read_from_file("", 1, 1)

    assert (
        "Seismic response coordinates with approximate locations "
        f"[(({east[0]}, {north[0]}), ({east[1]}, {north[1]}))] "
        "fall inside of a tolerance radius." in str(err.value)
    )
