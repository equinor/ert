from ert.config.analysis_iter_config import AnalysisIterConfig


def test_analysis_iter_config_default():
    c_default = AnalysisIterConfig()
    assert repr(c_default).startswith("AnalysisIterConfig")
    assert c_default.iter_case is None
    assert c_default.iter_count == 4
    assert c_default.iter_retry_count == 4


def test_analysis_iter_config_dict_init():
    iter_case = "new_name"
    iter_count = 42
    iter_retry_count = 24
    config_dict = {
        "ITER_CASE": iter_case,
        "ITER_COUNT": iter_count,
        "ITER_RETRY_COUNT": iter_retry_count,
    }
    a_iter_config = AnalysisIterConfig(**config_dict)

    assert a_iter_config.iter_case == iter_case
    assert a_iter_config.iter_count == iter_count
    assert a_iter_config.iter_retry_count == iter_retry_count
