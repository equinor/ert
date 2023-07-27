import os

from hypothesis import assume, given

from ert._c_wrappers.enkf import ErtConfig
from ert.parsing import ConfigKeys

from .config_dict_generator import config_generators


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
