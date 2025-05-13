import logging
import os

import pytest
from hypothesis import assume, given, settings

from ert.config import ErtConfig
from ert.config.ert_config import _substitutions_from_dict
from ert.config.parsing import ConfigKeys
from ert.substitutions import Substitutions

from .config.config_dict_generator import config_generators


@pytest.mark.integration_test
@settings(max_examples=10)
@given(config_generators(), config_generators())
def test_different_defines_give_different_subst_lists(
    tmp_path_factory, config_generator1, config_generator2
):
    with config_generator1(tmp_path_factory) as config_values1:
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


def test_subst_list_reads_correct_values():
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


def test_substitutions_with_hybrid_parameter_types():
    subst_list = Substitutions()
    params: dict[str, dict[str, float | str]] = {
        "GROUP1": {"a": 1.01},
        "GROUP2": {"b": "value"},
    }
    to_substitute = "<a> and <b>"
    assert subst_list.substitute_parameters(to_substitute, params) == "1.01 and value"


def test_substitutions():
    subst_list = Substitutions()

    subst_list["<Key>"] = "Value"

    assert len(subst_list) == 1

    with pytest.raises(KeyError):
        _ = subst_list["NoSuchKey"]

    assert "<Key>" in subst_list
    assert subst_list["<Key>"], "Value"

    subst_list["<Key2>"] = "Value2"
    assert list(subst_list.keys()) == ["<Key>", "<Key2>"]

    str_repr = repr(subst_list)
    assert "Substitutions" in str_repr
    assert "<Key2>, Value2" in str_repr
    assert "<Key>, Value" in str_repr

    assert subst_list.get("nosuchkey", 1729) == 1729
    assert subst_list.get("nosuchkey") is None
    assert subst_list.get(513) is None
    assert subst_list == {"<Key>": "Value", "<Key2>": "Value2"}


def test_substitutions_of_parameters_logs_warning_on_overlap_with_userconfig_values(
    caplog,
):
    caplog.set_level(logging.WARNING)
    to_substitute = "<my_key> and <CWD> and <new_key>"
    # first we apply the user config values (and CWD from magic string)
    config_dict = {
        "DEFINE": [
            ["<my_key>", "my_value_from_config"],
            ["<CWD>", "/my/path/somewhere"],  # Is a predefined key/magic string
        ],
    }
    subst_list = _substitutions_from_dict(config_dict)
    to_substitute = subst_list.substitute(to_substitute)

    overlapping_parameter_names: dict[str, dict[str, float | str]] = {
        "GROUP1": {"my_key": "ignored_value0"},
        "GROUP2": {"CWD": "ignored_value1"},
    }
    params = overlapping_parameter_names.copy()
    params["GROUP3"] = {"new_key": "new_value_from_param"}

    assert (
        subst_list.substitute_parameters(to_substitute, params)
        == "my_value_from_config and /my/path/somewhere and new_value_from_param"
    )  # Should not update if already set by user config
    for group_values in overlapping_parameter_names.values():
        for key, value in group_values.items():
            assert (
                f"Tried to substitute <{key}> for '{value}' from parameters, but it "
                f"was already set to '{subst_list.get(f'<{key}>')}' from user config"
                in caplog.text
            )
