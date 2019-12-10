import sys
import pandas as pd
import pytest

from ert_data import loader
from ert_data.measured import MeasuredData

if sys.version_info >= (3, 3):
    from unittest.mock import Mock
else:
    from mock import Mock


@pytest.fixture()
def valid_dataframe():
    return pd.DataFrame(data=[[1, 2, 3], [4, 5, 6]], index=["OBS", "STD"])


@pytest.fixture()
def measured_data_setup():
    def _setup(input_dataframe, monkeypatch):
        mocked_loader = Mock(return_value=input_dataframe)
        factory = Mock(return_value=mocked_loader)
        monkeypatch.setattr(loader, "data_loader_factory", factory)
        return factory

    return _setup


def _set_multiindex(df):
    tuples = list(zip(*[df.columns.to_list(), df.columns.to_list()]))
    return pd.MultiIndex.from_tuples(tuples, names=["key_index", "data_index"])

@pytest.mark.usefixtures("facade", "valid_dataframe", "measured_data_setup")
@pytest.mark.parametrize("obs_type", [("GEN_OBS"), ("SUMMARY_OBS"), ("BLOCK_OBS")])
def test_get_data(obs_type, monkeypatch, facade, valid_dataframe, measured_data_setup):

    facade.get_impl_type_name_for_obs_key.return_value = obs_type

    factory = measured_data_setup(valid_dataframe, monkeypatch)
    md = MeasuredData(facade, ["test_key"], index_lists=[[1, 2]])

    factory.assert_called_once_with(obs_type)
    mocked_loader = factory()
    mocked_loader.assert_called_once_with(facade, "test_key", "test_case")

    df = pd.DataFrame(data=[[2, 3], [5, 6]], index=["OBS", "STD"], columns=[1, 2])
    df.columns = _set_multiindex(df)
    expected_result = pd.concat({"test_key": df}, axis=1)

    assert md._data.equals(expected_result)


@pytest.mark.usefixtures("facade", "valid_dataframe", "measured_data_setup")
@pytest.mark.parametrize(
    "invalid_input,expected_error",
    [
        (1, TypeError),
        ("a", TypeError),
        ({1: 2}, TypeError),
        ([1, 2], TypeError),
        (pd.DataFrame(data=[1], index=["OBS"]), ValueError),
        (pd.DataFrame(data=[1], index=["not_expected"]), ValueError),
    ],
)
def test_invalid_set_data(
    facade,
    monkeypatch,
    invalid_input,
    expected_error,
    valid_dataframe,
    measured_data_setup,
):

    measured_data_setup(valid_dataframe, monkeypatch)
    md = MeasuredData(facade, ["test_key"], index_lists=[[1, 2]])

    with pytest.raises(expected_error):
        md._set_data(invalid_input)


@pytest.mark.usefixtures("facade", "measured_data_setup")
@pytest.mark.parametrize(
    "input_dataframe,expected_result",
    [
        (
            pd.DataFrame(
                data=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], index=["OBS", "STD", 1]
            ),
            pd.DataFrame(
                data=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], index=["OBS", "STD", 1]
            ),
        ),
        (
            pd.DataFrame(
                data=[[1, 2, 3], [4, 5, 6], [None, None, None]], index=["OBS", "STD", 1]
            ),
            pd.DataFrame(data=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], index=["OBS", "STD"]),
        ),
        (
            pd.DataFrame(
                data=[[1, 2, 3], [4, 5, None], [7, 8, None]], index=["OBS", "STD", 1]
            ),
            pd.DataFrame(
                data=[[1, 2, 3], [4, 5, None], [7, 8, None]], index=["OBS", "STD", 1]
            ),
        ),
    ],
)
def test_remove_failed_realizations(
    input_dataframe, expected_result, monkeypatch, facade, measured_data_setup
):
    measured_data_setup(input_dataframe, monkeypatch)
    md = MeasuredData(facade, ["test_key"])

    md.remove_failed_realizations()

    expected_result.columns = _set_multiindex(expected_result)

    assert md.data.equals(pd.concat({"test_key": expected_result}, axis=1))


@pytest.mark.usefixtures("facade", "measured_data_setup")
@pytest.mark.parametrize(
    "input_dataframe,expected_result",
    [
        (
            pd.DataFrame(
                data=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], index=["OBS", "STD", 1]
            ),
            pd.DataFrame(
                data=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], index=["OBS", "STD", 1]
            ),
        ),
        (
            pd.DataFrame(
                data=[[1, None, 3], [4, None, 6], [7, 8, 9]], index=["OBS", "STD", 1]
            ),
            pd.DataFrame(
                data=[[1, 3], [4, 6], [7, 9]], index=["OBS", "STD", 1], columns=[0, 2]
            ),
        ),
    ],
)
def test_remove_inactive_observations(
    input_dataframe, expected_result, monkeypatch, facade, measured_data_setup
):

    measured_data_setup(input_dataframe, monkeypatch)
    md = MeasuredData(facade, ["test_key"])

    expected_result.columns = _set_multiindex(expected_result)

    md.remove_inactive_observations()
    assert md.data.equals(pd.concat({"test_key": expected_result}, axis=1))


@pytest.mark.usefixtures("facade", "measured_data_setup")
@pytest.mark.parametrize(
    "std_cutoff,expected_result",
    [
        (
            -1,
            pd.DataFrame(
                data=[[1, 2], [0.1, 0.2], [1, 1.5], [1, 2.5]],
                index=["OBS", "STD", 1, 2],
            ),
        ),
        (
            0,
            pd.DataFrame(
                data=[[2], [0.2], [1.5], [2.5]], index=["OBS", "STD", 1, 2], columns=[1]
            ),
        ),
        (1.0, pd.DataFrame(index=["OBS", "STD", 1, 2])),
    ],
)
def test_filter_ensamble_std(
    std_cutoff, expected_result, monkeypatch, facade, measured_data_setup
):
    expected_result.columns = _set_multiindex(expected_result)

    input_dataframe = pd.DataFrame(
        data=[[1, 2], [0.1, 0.2], [1, 1.5], [1, 2.5]], index=["OBS", "STD", 1, 2]
    )

    measured_data_setup(input_dataframe, monkeypatch)
    md = MeasuredData(facade, ["test_key"])

    md.filter_ensemble_std(std_cutoff)
    assert md.data.equals(pd.concat({"test_key": expected_result}, axis=1))


@pytest.mark.usefixtures("facade", "measured_data_setup")
@pytest.mark.parametrize(
    "alpha,expected_result",
    [
        (
            10,
            pd.DataFrame(
                data=[[1, 2], [0.1, 0.2], [1.1, 1.6], [1, 2.5]],
                index=["OBS", "STD", 1, 2],
            ),
        ),
        (
            0.2,
            pd.DataFrame(
                data=[[2.0], [0.2], [1.6], [2.5]],
                index=["OBS", "STD", 1, 2],
                columns=[1],
            ),
        ),
        (0, pd.DataFrame(index=["OBS", "STD", 1, 2])),
    ],
)
def test_filter_ens_mean_obs(
    alpha, expected_result, monkeypatch, facade, measured_data_setup
):
    expected_result.columns = _set_multiindex(expected_result)

    input_dataframe = pd.DataFrame(
        data=[[1, 2], [0.1, 0.2], [1.1, 1.6], [1, 2.5]], index=["OBS", "STD", 1, 2]
    )

    measured_data_setup(input_dataframe, monkeypatch)
    md = MeasuredData(facade, ["test_key"])

    md.filter_ensemble_mean_obs(alpha)
    assert md.data.equals(pd.concat({"test_key": expected_result}, axis=1))


@pytest.mark.parametrize(
    "index_list,expected_result",
    [
        (None, pd.DataFrame(data=[[1, 2, 3, 4, 5, 6, 7, 8]])),
        ([0], pd.DataFrame(data=[[1]])),
        ([5, 6], pd.DataFrame(data=[[6, 7]], columns=[5, 6])),
        ([0, 1, 2, 3, 4, 5, 6, 7], pd.DataFrame(data=[[1, 2, 3, 4, 5, 6, 7, 8]])),
    ],
)
def test_filter_on_column_index(index_list, expected_result):
    input_data = pd.DataFrame(data=[[1, 2, 3, 4, 5, 6, 7, 8]])
    result = MeasuredData._filter_on_column_index(input_data, index_list)
    assert result.equals(expected_result)


@pytest.mark.usefixtures("facade", "measured_data_setup")
@pytest.mark.parametrize(
    "input_dataframe,expected_result",
    [
        (
            pd.DataFrame(
                data=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], index=["OBS", "STD", 1]
            ),
            pd.DataFrame(
                data=[[7, 8, 9]], index=[1]
            ),
        ),
        (
            pd.DataFrame(
                data=[[1, None, 3], [4, None, 6], [7, 8, 9], [10, 11, 12]], index=["OBS", "STD", 1, 2]
            ),
            pd.DataFrame(
                data=[[7, 8, 9], [10, 11, 12]], index=[1, 2], columns=[0, 1, 2]
            ),
        ),
    ],
)
def test_get_simulated_data(
    input_dataframe, expected_result, monkeypatch, facade, measured_data_setup
):

    measured_data_setup(input_dataframe, monkeypatch)
    md = MeasuredData(facade, ["test_key"])

    expected_result.columns = _set_multiindex(expected_result)

    result = md.get_simulated_data()
    assert result.equals(pd.concat({"test_key": expected_result.astype(float)}, axis=1))