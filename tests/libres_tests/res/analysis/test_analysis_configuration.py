import pytest

from res.enkf import LocalConfig, LocalObsdata
from res.enkf.local_ministep import Observation, LocalMinistep
from res._lib.update import RowScalingParameter, Parameter
from res.enkf.row_scaling import RowScaling


def test_observation_default_init():
    observation = Observation("Obs_name")
    assert observation.active_index == []


def test_parameter_default_init():
    parameter = Parameter("Parameter_name")
    assert parameter.active_list.getMode().name == "ALL_ACTIVE"
    assert parameter.index_list == []


@pytest.mark.parametrize(
    "input_list, expected_mode", [([], "ALL_ACTIVE"), ([1], "PARTLY_ACTIVE")]
)
def test_parameter(input_list, expected_mode):
    parameter = Parameter("Parameter_name", input_list)
    assert parameter.active_list.getMode().name == expected_mode
    assert parameter.index_list == input_list
    assert parameter.active_list.get_active_index_list() == input_list


def test_parameter_reset_active():
    parameter = Parameter("Parameter_name", [1, 2])
    assert parameter.active_list.getMode().name == "PARTLY_ACTIVE"
    parameter.index_list = []
    assert parameter.active_list.getMode().name == "ALL_ACTIVE"


def test_row_scaling_parameter_default_init():
    parameter = RowScalingParameter("Parameter_name", RowScaling())
    assert parameter.active_list.getMode().name == "ALL_ACTIVE"
    assert parameter.index_list == []


def test_configuration():
    config = LocalConfig()
    ministep = config.add_ministep("ministep_name")
    ministep.add_observation("MY_OBS")
    ministep.add_parameter("MY_PARAMETER")
    ministep.add_row_scaling_parameter("MY_ROW_SCALING", RowScaling())

    config.context_validate(["MY_OBS"], ["MY_PARAMETER", "MY_ROW_SCALING"])


@pytest.mark.parametrize(
    "input_obs", [["OBS"], [["OBS", [1, 2, 3]]], ["OBS_1", ["OBS", [1, 2, 3]]]]
)
def test_configuration_valid_obs_input(input_obs):
    config = LocalConfig()
    config.add_ministep(
        "ministep_name",
        observations=input_obs,
        parameters=["MY_PARAMETER", "MY_ROW_SCALING"],
    )

    config.context_validate(["OBS"], ["MY_PARAMETER", "MY_ROW_SCALING"])
