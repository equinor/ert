from ert._c_wrappers.enkf import AnalysisIterConfig


def test_set_analysis_iter_config():
    c = AnalysisIterConfig()
    assert repr(c).startswith("AnalysisIterConfig")

    assert not c.caseFormatSet()
    c.setCaseFormat("case%d")
    assert c.caseFormatSet()

    assert not c.numIterationsSet()
    c.setNumIterations(1)
    assert c.numIterationsSet()


def test_analysis_iter_config_default():
    c_default = AnalysisIterConfig()
    c_dict = AnalysisIterConfig(config_dict={})
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
    a_ite_config = AnalysisIterConfig(config_dict=config_dict)

    assert a_ite_config.getCaseFormat() == iter_case
    assert a_ite_config.getNumIterations() == iter_count
    assert a_ite_config.getNumRetries() == iter_retry_count
