from pathlib import Path
from textwrap import dedent
from typing import cast

import pytest

from ert.config._create_observation_dataframes import _handle_seismic_observation
from ert.config._observations import SeismicObservation
from ert.config.ert_config import ErtConfig
from ert.config.response_config import InvalidResponseFile
from ert.config.seismic_config import SeismicConfig
from tests.ert.defaults_generator import create_seismic_observation_dict


@pytest.mark.usefixtures("use_tmpdir")
def test_that_seismic_observation_response_key_matches_simulated_response_key():
    expected_response_key = "horizon--amplitude_full_min_depth--20250101_20240101"
    name = f"{expected_response_key}.csv"
    runpath = "runpath"
    obs_path = "share/preprocessed/tables/" + name
    simulated_path_relative_to_runpath = "share/results/tables/" + name
    simulated_path = runpath + "/" + simulated_path_relative_to_runpath

    obs_content = dedent(
        """
        X_UTME,Y_UTMN,OBS,OBS_ERROR,REGION
        100.00,200.00,1.0,0.005,1.0
        105.00,205.00,2.0,0.005,1.0
        """
    )

    simulated_content = dedent(
        """
        X_UTME,Y_UTMN,OBS,OBS_ERROR,REGION
        100.00,200.00,1.1,0.005,1.0
        105.00,205.00,2.2,0.005,1.0
        """
    )
    Path(obs_path).parent.mkdir(parents=True, exist_ok=True)
    Path(simulated_path).parent.mkdir(parents=True, exist_ok=True)
    Path(obs_path).write_text(obs_content, encoding="utf8")
    Path(simulated_path).write_text(simulated_content, encoding="utf8")

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
        seismic_config.read_from_file(tmp_path, 1, 1)


def test_that_seismic_config_reads_from_all_input_files(mocked_files):
    key1 = "horizon--amplitude_full_min_depth--20250101_20240101"
    key2 = "horizon--amplitude_full_mean_depth--20260101_20240101"
    name1 = f"{key1}.csv"
    name2 = f"{key2}.csv"

    mocked_files[name1] = dedent(
        """
        X_UTME,Y_UTMN,OBS,OBS_ERROR,REGION
        100.00,200.00,1.0,0.005,1.0
        105.00,205.00,2.0,0.005,1.0
        """
    )

    mocked_files[name2] = dedent(
        """
        X_UTME,Y_UTMN,OBS,OBS_ERROR,REGION
        100.00,200.00,3.0,0.005,1.0
        105.00,205.00,4.0,0.005,1.0
        """
    )

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
def test_that_seismic_config_supports_blob_pattern():
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


def test_that_empty_seismic_response_file_does_not_raise(mocked_files):
    key = "horizon--amplitude_full_min_depth--20250101_20240101"
    name = f"{key}.csv"

    mocked_files[name] = dedent(
        """
        X_UTME,Y_UTMN,OBS,OBS_ERROR,REGION
        """
    )

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
@pytest.mark.usefixtures("use_tmpdir")
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
