import os

import pytest
from hypothesis import assume, given, settings

from ert.config import ErtConfig
from ert.config.parsing import ConfigKeys
from ert.substitution_list import SubstitutionList

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
                ert_config1.substitution_list
                != ErtConfig.from_dict(
                    config_values2.to_config_dict("test.ert", os.getcwd())
                ).substitution_list
            )


def test_subst_list_reads_correct_values():
    substitution_list = ErtConfig.from_dict(
        {
            ConfigKeys.NUM_REALIZATIONS: 1,
            ConfigKeys.DEFINE: [
                ("keyA", "valA"),
                ("keyB", "valB"),
            ],
            ConfigKeys.DATA_KW: [("keyC", "valC"), ("keyD", "valD")],
            ConfigKeys.ENSPATH: "test",
        }
    ).substitution_list
    assert substitution_list["keyA"] == "valA"
    assert substitution_list["keyB"] == "valB"
    assert substitution_list["keyC"] == "valC"
    assert substitution_list["keyD"] == "valD"


def test_substitution_list():
    subst_list = SubstitutionList()

    subst_list["<Key>"] = "Value"

    assert len(subst_list) == 1

    with pytest.raises(KeyError):
        _ = subst_list["NoSuchKey"]

    assert "<Key>" in subst_list
    assert subst_list["<Key>"], "Value"

    subst_list["<Key2>"] = "Value2"
    assert list(subst_list.keys()) == ["<Key>", "<Key2>"]

    str_repr = repr(subst_list)
    assert "SubstitutionList" in str_repr
    assert "<Key2>, Value2" in str_repr
    assert "<Key>, Value" in str_repr

    assert subst_list.get("nosuchkey", 1729) == 1729
    assert subst_list.get("nosuchkey") is None
    assert subst_list.get(513) is None
    assert subst_list == {"<Key>": "Value", "<Key2>": "Value2"}
