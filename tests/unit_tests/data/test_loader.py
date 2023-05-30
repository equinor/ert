from unittest.mock import MagicMock, Mock

import numpy as np
import pandas as pd
import pytest

from ert._c_wrappers.enkf import GenObservation
from ert.data import loader


def create_summary_get_observations():
    return pd.DataFrame(
        [[10.0, None, 10.0, 10.0], [1.0, None, 1.0, 1.0]], index=["OBS", "STD"]
    )


@pytest.mark.parametrize(
    "obs_type",
    ["GEN_OBS", "SUMMARY_OBS"],
)
def test_data_loader_factory(obs_type):
    # pylint: disable=comparison-with-callable, no-member
    # (pylint is not fully understanding the .func here)
    assert loader.data_loader_factory(obs_type).func == loader._extract_data


def test_data_loader_factory_fails():
    with pytest.raises(TypeError):
        loader.data_loader_factory("BAD_TYPE")


def test_load_general_response(facade, new_ensemble):
    facade.load_gen_data.return_value = pd.DataFrame(data=[10.0, 10.0, 10.0, 10.0])
    facade.all_data_type_keys.return_value = ["some_key@1"]
    facade.is_gen_data_key.return_value = True

    result = loader._load_general_response(facade, new_ensemble, "some_key")

    facade.load_gen_data.assert_called_once_with(new_ensemble, "some_key", 1)

    assert result.equals(
        pd.DataFrame(
            [[10.0, 10.0, 10.0, 10.0]],
            index=[0],
        )
    )


def test_load_general_obs(facade):
    facade.get_observations.return_value = {
        "some_key": [
            GenObservation(
                np.array([10.0, 20.0, 30.0]),
                np.array([1.0, 2.0, 3.0]),
                np.array([0, 2, 3]),
                np.full(3, 1.0),
            )
        ]
    }

    facade.load_gen_data.return_value = pd.DataFrame(data=[9.9, 19.9, 29.9, 39.9])

    result = loader._load_general_obs(facade, Mock(), ["some_key"])

    assert result.columns.to_list() == [
        ("some_key", 0, 0),
        ("some_key", 2, 2),
        ("some_key", 3, 3),
    ]
    assert result.index.to_list() == ["OBS", "STD"]
    assert all(result.values.flatten() == [10.0, 20.0, 30.0, 1.0, 2.0, 3.0])


@pytest.mark.parametrize("func", [loader.data_loader_factory("SUMMARY_OBS")])
def test_load_summary_data(facade, monkeypatch, func):
    obs_mock = Mock()
    obs_mock.observations = {1: Mock(), 2: Mock()}

    facade.get_observations.return_value = {"some_key": obs_mock}
    facade.load_observation_data.return_value = pd.DataFrame(
        {
            "some_key": {"2010-01-10": 10, "2010-01-20": 20},
            "STD_some_key": {"2010-01-10": 0.1, "2010-01-20": 0.2},
        }
    )
    facade.load_all_summary_data.return_value = pd.DataFrame(
        {"some_key": [9.9, 19.9, 29.9]},
        index=pd.MultiIndex.from_tuples(
            [(0, "2010-01-10"), (0, "2010-01-20"), (0, "2021-02-10")],
            names=["Realization", "Date"],
        ),
    )
    facade.get_impl_type_name_for_obs_key.return_value = "SUMMARY_OBS"

    result = func(facade, Mock(), "some_key")

    assert result.columns.to_list() == [
        ("some_key", "2010-01-10", 0),
        ("some_key", "2010-01-20", 1),
    ]
    assert result.index.to_list() == ["OBS", "STD", 0]
    assert all(result.values.flatten() == [10.0, 20.0, 0.1, 0.2, 9.9, 19.9])


def test_load_summary_obs(facade, monkeypatch):
    obs_mock = Mock()
    obs_mock.observations = {1: Mock(), 2: Mock()}

    facade.get_observations.return_value = {"some_key": obs_mock}
    facade.load_observation_data.return_value = pd.DataFrame(
        {
            "some_key": {"2010-01-10": 10, "2010-01-20": 20},
            "STD_some_key": {"2010-01-10": 0.1, "2010-01-20": 0.2},
        }
    )

    result = loader._load_summary_obs(facade, Mock(), ["some_key"])

    assert result.columns.to_list() == [
        ("some_key", "2010-01-10", 0),
        ("some_key", "2010-01-20", 1),
    ]
    assert result.index.to_list() == ["OBS", "STD"]
    assert all(result.values.flatten() == [10.0, 20.0, 0.1, 0.2])


def test_load_summary_response(facade, monkeypatch):
    facade.load_all_summary_data.return_value = pd.DataFrame(
        {"some_key": [9.9, 19.9, 29.9]},
        index=pd.MultiIndex.from_tuples(
            [(0, "2010-01-10"), (0, "2010-01-20"), (0, "2021-02-10")],
            names=["Realization", "Date"],
        ),
    )

    result = loader._load_summary_response(facade, "some_key", "a_random_name")

    assert result.columns.to_list() == ["2010-01-10", "2010-01-20", "2021-02-10"]
    assert result.index.to_list() == [0]
    assert all(result.values.flatten() == [9.9, 19.9, 29.9])


def test_no_obs_error(facade, monkeypatch):
    obs_mock = pd.DataFrame()
    obs_loader = Mock(return_value=obs_mock)
    response_loader = Mock(return_value=pd.DataFrame([1, 2]))
    monkeypatch.setattr(loader, "_create_multi_index", MagicMock(return_value=[1]))
    facade.get_impl_type_name_for_obs_key.return_value = "SUMMARY_OBS"
    with pytest.raises(loader.ObservationError):
        loader._extract_data(
            facade,
            Mock(),
            "some_key",
            response_loader,
            obs_loader,
            "SUMMARY_OBS",
        )


def test_that_its_not_possible_to_load_multiple_different_obs_types(facade):
    facade.get_impl_type_name_for_obs_key.side_effect = [
        "SUMMARY_OBS",
        "GEN_OBS",
    ]
    with pytest.raises(
        loader.ObservationError,
        match=(
            r"Expected only SUMMARY_OBS observation type. "
            r"Found: \['SUMMARY_OBS', 'GEN_OBS'\] for "
            r"\['some_summary_key', 'block_key'\]"
        ),
    ):
        loader._extract_data(
            facade,
            Mock(),
            ["some_summary_key", "block_key"],
            Mock(),
            Mock(),
            "SUMMARY_OBS",
        )


def test_different_data_key(facade):
    facade.get_impl_type_name_for_obs_key.return_value = "SUMMARY_OBS"
    facade.get_data_key_for_obs_key.side_effect = [
        "data_key",
        "another_data_key",
        "data_key",
    ]
    with pytest.raises(
        loader.ObservationError,
        match=r"found: \['data_key', 'another_data_key', 'data_key'\]",
    ):
        loader._extract_data(
            facade,
            Mock(),
            ["obs_1", "obs_2", "obs_3"],
            "a_random_name",
            Mock(),
            Mock(),
            "SUMMARY_OBS",
        )
