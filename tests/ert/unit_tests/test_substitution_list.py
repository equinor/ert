import os

import pytest
from hypothesis import assume, given, settings

from ert.config import ErtConfig
from ert.config.parsing import ConfigKeys
from ert.substitutions import Substitutions

from .config.config_dict_generator import config_generators


@pytest.mark.integration_test
@pytest.mark.filterwarnings("ignore:MIN_REALIZATIONS")
@pytest.mark.filterwarnings("ignore:Config contains a SUMMARY key")
@pytest.mark.filterwarnings("ignore:Duplicate forward model step")
@settings(max_examples=100)
@given(config_generators(), config_generators())
def test_different_defines_give_different_subst_lists(
    tmp_path_factory, config_generator1, config_generator2
):
    with (
        config_generator1(tmp_path_factory) as config_values1,
    ):
        ert_config1 = ErtConfig.from_dict(
            config_values1.to_config_dict("test.ert", os.getcwd())
        )
        with config_generator2(tmp_path_factory) as config_values2:
            assume(config_values1.define != config_values2.define)
            assert (
                ert_config1.substitutions
                != ErtConfig.from_dict(
                    config_values2.to_config_dict("test.ert", os.getcwd())
                ).substitutions
            )


def test_that_define_and_data_kw_parameters_are_used_as_substitutions():
    substitutions = ErtConfig.from_dict(
        {
            ConfigKeys.NUM_REALIZATIONS: 1,
            ConfigKeys.DEFINE: [
                ("keyA", "valA"),
                ("keyB", "valB"),
            ],
            ConfigKeys.DATA_KW: [("keyC", "valC"), ("keyD", "valD")],
            ConfigKeys.ENSPATH: "test",
        }
    ).substitutions
    assert substitutions["keyA"] == "valA"
    assert substitutions["keyB"] == "valB"
    assert substitutions["keyC"] == "valC"
    assert substitutions["keyD"] == "valD"


def test_that_parameters_as_magic_strings_are_substituted():
    params: dict[str, dict[str, float | str]] = {
        "GROUP1": {"a": 1.01},
        "GROUP2": {"b": "value"},
    }
    to_substitute = "<a> and <b>"
    assert (
        Substitutions.substitute_parameters(to_substitute, params) == "1.01 and value"
    )
