from ert._c_wrappers.enkf import AnalysisIterConfig


def test_set_analysis_iter_config():
    c = AnalysisIterConfig()
    assert repr(c).startswith("AnalysisIterConfig")

    assert not c.case_format_is_set()
    c.set_case_format("case%d")
    assert c.case_format_is_set()


def test_analysis_iter_config_default():
    c_default = AnalysisIterConfig()
    c_dict = AnalysisIterConfig.from_dict({})
    assert c_default == c_dict


def test_analysis_iter_config_dict_init():
    iter_case = "new_name"
    iter_count = 42
    iter_retry_count = 24
    config_dict = {
        "ITER_CASE": iter_case,
        "ITER_COUNT": iter_count,
        "ITER_RETRY_COUNT": iter_retry_count,
    }
    a_ite_config = AnalysisIterConfig.from_dict(config_dict)

    assert a_ite_config.case_format() == iter_case
    assert a_ite_config.get_num_iterations() == iter_count
    assert a_ite_config.get_num_retries() == iter_retry_count
