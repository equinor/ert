from textwrap import dedent
from typing import cast

import pytest

from ert.config._create_observation_dataframes import _handle_seismic_observation
from ert.config._observations import SeismicObservation
from ert.config.ert_config import ErtConfig
from ert.config.parsing.observations_parser import ObservationType
from ert.config.response_config import InvalidResponseFile
from ert.config.seismic_config import SeismicConfig


def test_that_seismic_observation_response_key_matches_simulated_response_key(
    mocked_files,
):
    expected_response_key = "field--amplitude_full_min_depth--20250101_20240101"
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
                    {
                        "type": ObservationType.SEISMIC,
                        "name": "OBS1",
                        "CSV": obs_path,
                    },
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


def test_that_seismic_config_reads_from_all_input_files(mocked_files):
    key1 = "field--amplitude_full_min_depth--20250101_20240101"
    key2 = "field--amplitude_full_mean_depth--20260101_20240101"
    name1 = f"{key1}.csv"
    name2 = f"{key2}.csv"
    runpath = "/runpath"
    simulated_path1 = runpath + "/" + name1
    simulated_path2 = runpath + "/" + name2

    mocked_files[simulated_path1] = dedent(
        """
        X_UTME,Y_UTMN,OBS,OBS_ERROR,REGION
        100.00,200.00,1.0,0.005,1.0
        105.00,205.00,2.0,0.005,1.0
        """
    )

    mocked_files[simulated_path2] = dedent(
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

    data = seismic_config.read_from_file(runpath, 1, 1)
    assert data.shape == (4, 4)
    assert data["response_key"].to_list() == [key1, key1, key2, key2]
    assert data["east"].to_list() == [100.0, 105.0, 100.0, 105.0]
    assert data["north"].to_list() == [200.0, 205.0, 200.0, 205.0]
    assert data["values"].to_list() == [1.0, 2.0, 3.0, 4.0]


@pytest.mark.parametrize(
    ("east", "north"),
    [
        pytest.param([111.11, 111.11], [222.22, 222.22], id="same coordinates"),
        pytest.param(
            [111.1111111111111111111111, 111.11111111111111],
            [222.2222222222222222222222222222222, 222.22222222222223],
            id="lost precision",
        ),
    ],
)
@pytest.mark.usefixtures("use_tmpdir")
def test_that_duplicate_location_in_seismic_response_raises(mocked_files, east, north):
    key = "field--amplitude_full_min_depth--20250101_20240101"
    name = f"{key}.csv"
    runpath = "/runpath"
    simulated_path = runpath + "/" + name

    mocked_files[simulated_path] = dedent(
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
        seismic_config.read_from_file(runpath, 1, 1)

    assert "Seismic response coordinates were not unique (after rounding)" in str(
        err.value
    )
