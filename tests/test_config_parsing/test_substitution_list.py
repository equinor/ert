from hypothesis import assume, given

from ert._c_wrappers.enkf import ConfigKeys, ResConfig

from .config_dict_generator import config_generators


@given(config_generators(), config_generators())
def test_different_defines_give_different_subst_lists(
    tmp_path_factory, config_generator1, config_generator2
):
    with config_generator1(tmp_path_factory) as config_dict1:
        res_config1 = ResConfig(config_dict=config_dict1)
        with config_generator2(tmp_path_factory) as config_dict2:
            assume(
                config_dict1[ConfigKeys.DEFINE_KEY]
                != config_dict2[ConfigKeys.DEFINE_KEY]
            )
            assert (
                res_config1.substitution_list
                != ResConfig(config_dict=config_dict2).substitution_list
            )


@given(config_generators())
def test_from_dict_and_from_file_creates_equal_subst_lists(
    tmp_path_factory, config_generator
):
    filename = "config.ert"
    with config_generator(tmp_path_factory, filename) as config_dict:
        res_config_from_file = ResConfig(user_config_file=filename)
        res_config_from_dict = ResConfig(config_dict=config_dict)
        assert (
            res_config_from_dict.substitution_list
            == res_config_from_file.substitution_list
        )


def test_subst_list_reads_correct_values():
    substitution_list = ResConfig(
        config_dict={
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
