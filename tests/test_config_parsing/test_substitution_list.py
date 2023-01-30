from hypothesis import assume, given

from ert._c_wrappers.enkf import ConfigKeys, ErtConfig

from .config_dict_generator import config_generators


@given(config_generators(), config_generators())
def test_different_defines_give_different_subst_lists(
    tmp_path_factory, config_generator1, config_generator2
):
    with config_generator1(tmp_path_factory) as config_dict1:
        res_config1 = ErtConfig.from_dict(config_dict1)
        with config_generator2(tmp_path_factory) as config_dict2:
            assume(
                config_dict1[ConfigKeys.DEFINE_KEY]
                != config_dict2[ConfigKeys.DEFINE_KEY]
            )
            assert (
                res_config1.substitution_list
                != ErtConfig.from_dict(config_dict2).substitution_list
            )


def test_subst_list_reads_correct_values():
    substitution_list = ErtConfig.from_dict(
        {
            ConfigKeys.NUM_REALIZATIONS: 1,
            ConfigKeys.DEFINE_KEY: [
                ("keyA", "valA"),
                ("keyB", "valB"),
            ],
            ConfigKeys.DATA_KW_KEY: [("keyC", "valC"), ("keyD", "valD")],
            ConfigKeys.ENSPATH: "test",
        }
    ).substitution_list
    assert substitution_list["keyA"] == "valA"
    assert substitution_list["keyB"] == "valB"
    assert substitution_list["keyC"] == "valC"
    assert substitution_list["keyD"] == "valD"
